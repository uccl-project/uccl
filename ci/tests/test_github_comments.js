const assert = require('node:assert/strict');
const {test} = require('node:test');
const {dispatch, update} = require('../github_comments.js');

function fixture(command = '/run-ci l4 p2p.*', permission = 'write') {
  const calls = [];
  const sha = 'a'.repeat(40);
  const context = {
    repo: {owner: 'uccl-project', repo: 'uccl'}, runId: 100, eventName: 'issue_comment',
    serverUrl: 'https://github.com', apiUrl: 'https://api.github.com',
    payload: {
      issue: {number: 12, pull_request: {}}, repository: {default_branch: 'main'},
      comment: {body: command, user: {login: 'maintainer'}, html_url: 'https://github.com/uccl-project/uccl/pull/12#issuecomment-10'},
    },
  };
  const pr = {number: 12, state: 'open', head: {sha, repo: {full_name: 'contributor/uccl'}}};
  const reply = {
    id: 200, user: {login: 'github-actions[bot]', type: 'Bot'},
    issue_url: 'https://api.github.com/repos/uccl-project/uccl/issues/12',
    html_url: 'https://github.com/uccl-project/uccl/pull/12#issuecomment-200',
  };
  let inputs;
  const github = {rest: {
    repos: {getCollaboratorPermissionLevel: async args => {
      calls.push(['permission', args]); return {data: {permission}};
    }},
    pulls: {get: async args => {calls.push(['pull', args]); return {data: pr};}},
    issues: {
      createComment: async args => {calls.push(['create', args]); reply.body = args.body; return {data: reply};},
      getComment: async args => {calls.push(['get', args]); return {data: reply};},
      updateComment: async args => {calls.push(['update', args]); reply.body = args.body; return {data: reply};},
    },
    actions: {createWorkflowDispatch: async args => {calls.push(['dispatch', args]); inputs = args.inputs;}},
  }};
  const core = {summary: {addRaw() {}, async write() {}}, warning() {}};
  return {
    github, context, core, calls, pr, reply, attempt: '1',
    startRun() {
      context.eventName = 'workflow_dispatch';
      context.runId = 101;
      context.payload = {inputs};
    },
  };
}

function results(overrides = {}) {
  return Object.fromEntries(Object.entries({
    'comment-start': 'success', plan: 'success', cpu: 'skipped', GPU: 'success', status: 'success', ...overrides,
  }).map(([id, result]) => [id, {result}]));
}

test('both commands authorize against the base repo and dispatch the exact fork SHA', async () => {
  for (const command of ['/run-ci all', '/run-benchmark gh200 ep.bench.*,p2p.*']) {
    const f = fixture(command);
    await dispatch(f);
    assert.deepEqual(f.calls.map(([kind]) => kind), ['permission', 'pull', 'create', 'dispatch']);
    assert.deepEqual(f.calls[0][1], {owner: 'uccl-project', repo: 'uccl', username: 'maintainer'});
    const request = f.calls[3][1];
    assert.equal(request.ref, 'main');
    assert.equal(request.owner, 'uccl-project');
    assert.equal(request.repo, 'uccl');
    assert.equal(request.inputs.sha, f.pr.head.sha);
    assert.equal(request.inputs.head_repo, 'contributor/uccl');
    assert.equal(request.inputs.pr, '12');
    assert.equal(request.inputs.reply_comment_id, '200');
    assert.equal(request.inputs.reply_marker, 'uccl-ci-request-100-1');
    assert.equal(request.inputs.tiers, command.startsWith('/run-ci') ? 'cpu,build,integration' : 'performance');
    assert.equal(request.inputs.targets, command.startsWith('/run-ci') ? '' : 'gh200');
    assert.equal(request.inputs.tests, command.startsWith('/run-ci') ? '' : 'ep.bench.*,p2p.*');
    assert.match(f.reply.body, /Status: queued/);
  }
});

test('read, triage and unknown permissions cannot comment or dispatch', async () => {
  for (const permission of ['read', 'triage', 'none', undefined]) {
    const f = fixture('/run-ci l4', permission);
    if (permission === undefined) {
      f.github.rest.repos.getCollaboratorPermissionLevel = async () => ({data: {}});
    }
    await assert.rejects(dispatch(f), /Only maintainers/);
    assert.ok(f.calls.every(([kind]) => kind === 'permission'));
  }
});

test('all three maintainer permission levels retain access', async () => {
  for (const permission of ['write', 'maintain', 'admin']) {
    const f = fixture('/run-ci cpu', permission);
    await dispatch(f);
    assert.equal(f.calls.at(-1)[0], 'dispatch');
  }
});

test('invalid commands, ordinary issues and closed/deleted-source PRs do not dispatch', async () => {
  for (const command of ['/run-ci amd', '/run-benchmarks l4', '/run-ci l4; echo bad', '/run-ci l4\n/run-ci gh200']) {
    const f = fixture(command);
    await assert.rejects(dispatch(f), /Usage:/);
    assert.equal(f.calls.length, 0);
  }
  for (const state of ['issue', 'closed', 'deleted']) {
    const f = fixture();
    if (state === 'issue') delete f.context.payload.issue.pull_request;
    if (state === 'closed') f.pr.state = 'closed';
    if (state === 'deleted') f.pr.head.repo = null;
    await assert.rejects(dispatch(f));
    assert.ok(!f.calls.some(([kind]) => ['create', 'dispatch'].includes(kind)));
  }
});

test('dispatch failure updates the queued comment instead of leaving it queued', async () => {
  const f = fixture();
  f.github.rest.actions.createWorkflowDispatch = async () => {throw new Error('dispatch refused');};
  await assert.rejects(dispatch(f), /dispatch refused/);
  assert.equal(f.calls.filter(([kind]) => kind === 'create').length, 1);
  assert.equal(f.calls.at(-1)[1].comment_id, 200);
  assert.match(f.reply.body, /Status: failed to dispatch/);
});

test('queued, running and final status update one comment and preserve the original request', async () => {
  for (const command of ['/run-ci l4', '/run-benchmark gh200']) {
    const f = fixture(command);
    await dispatch(f);
    f.startRun();
    await update({...f, status: 'running'});
    assert.match(f.reply.body, /Status: running/);
    assert.match(f.reply.body, /actions\/runs\/101\/attempts\/1/);
    await update({...f, status: 'complete', results: results()});
    assert.match(f.reply.body, /Status: completed successfully/);
    assert.match(f.reply.body, /Requested by: @maintainer/);
    assert.ok(f.reply.body.includes(command));
    assert.ok(f.reply.body.includes(f.pr.head.sha));
    assert.match(f.reply.body, /\| CPU \| skipped \|/);
    assert.equal(f.calls.filter(([kind]) => kind === 'create').length, 1);
    assert.ok(f.calls.filter(([kind]) => kind === 'update').every(([, args]) => args.comment_id === 200));
    await update({...f, status: 'running', attempt: '2'});
    assert.match(f.reply.body, /actions\/runs\/101\/attempts\/2/);
  }
});

test('final status covers setup/test failures, cancellation and skipped execution', async () => {
  for (const [changes, label] of [
    [{plan: 'failure', cpu: 'skipped', GPU: 'skipped', status: 'failure'}, 'failed'],
    [{GPU: 'failure', status: 'failure'}, 'failed'],
    [{'comment-start': 'failure'}, 'failed'],
    [{GPU: 'cancelled', status: 'failure'}, 'cancelled'],
    [{cpu: 'skipped', GPU: 'skipped'}, 'skipped'],
  ]) {
    const f = fixture();
    await dispatch(f); f.startRun();
    await update({...f, status: 'complete', results: results(changes)});
    assert.ok(f.reply.body.includes(`**Status: ${label}**`));
  }
});

test('updates reject another author, PR, marker, SHA or repository', async () => {
  for (const change of [
    f => {f.reply.user.login = 'another-bot[bot]';},
    f => {f.reply.user.type = 'User';},
    f => {f.reply.issue_url += '3';},
    f => {f.context.payload.inputs.reply_marker = 'uccl-ci-request-999-1';},
    f => {f.context.payload.inputs.sha = 'b'.repeat(40);},
    f => {f.context.payload.inputs.head_repo = 'different/uccl';},
    f => {f.reply.body = 'unrelated comment';},
  ]) {
    const f = fixture();
    await dispatch(f); f.startRun(); change(f);
    await assert.rejects(update({...f, status: 'running'}), /does not match/);
    assert.equal(f.calls.filter(([kind]) => kind === 'update').length, 0);
  }
});

test('malformed inputs, PR events and missing results cannot report success', async () => {
  const f = fixture();
  await dispatch(f);
  await assert.rejects(update({...f, status: 'running'}), /dispatched run/);
  f.startRun();
  await assert.rejects(update({...f, status: 'complete', results: {}}), /Missing or invalid/);
  f.context.payload.inputs.reply_comment_id = 'invalid';
  await assert.rejects(update({...f, status: 'running'}), /Invalid CI reply identifiers/);
  assert.equal(f.calls.filter(([kind]) => kind === 'update').length, 0);
});
