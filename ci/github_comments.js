// Loaded only from the trusted workflow commit by jobs with comment permission.
const STATUS_MARKER = '\n<!-- uccl-ci-status -->\n';
const ALLOWED_PERMISSIONS = new Set(['write', 'maintain', 'admin']);

function runUrl(context, attempt) {
  const {owner, repo} = context.repo;
  return `${context.serverUrl || 'https://github.com'}/${owner}/${repo}/actions/runs/${context.runId}/attempts/${attempt}`;
}

function bodyWithStatus(header, status, url, results) {
  const lines = [header, `**Status: ${status}**`, '', `[Workflow run and logs](${url})`];
  if (results) {
    lines.push('', '| Job | Result |', '|---|---|');
    for (const [id, label] of [['plan', 'Plan'], ['cpu', 'CPU'], ['GPU', 'GPU'], ['status', 'CI status']]) {
      lines.push(`| ${label} | ${results[id].result} |`);
    }
    lines.push('', 'Per-suite results, skips and timings are in the run summary and artifacts.');
  }
  return lines.join('\n');
}

function completionStatus(results) {
  const states = ['plan', 'cpu', 'GPU', 'status'].map(id => results[id]?.result);
  if (states.some(state => !['success', 'failure', 'cancelled', 'skipped'].includes(state))) {
    throw new Error('Missing or invalid CI job results');
  }
  if (states.includes('cancelled')) return 'cancelled';
  if (states.includes('failure') || results['comment-start']?.result === 'failure') return 'failed';
  if (results.cpu.result === 'skipped' && results.GPU.result === 'skipped') return 'skipped';
  return 'completed successfully';
}

async function dispatch({github, context, core, attempt = process.env.GITHUB_RUN_ATTEMPT || '1'}) {
  const {owner, repo} = context.repo;
  if (!context.payload.issue?.pull_request) throw new Error('CI commands require a pull request');
  const command = context.payload.comment.body.trim();
  const match = command.match(/^\/(run-ci|run-benchmark)\s+(all|cpu|l4|gb10|gh200)(?:\s+([a-z0-9_.*,-]+))?$/);
  if (!match) throw new Error('Usage: /run-ci|/run-benchmark <all|cpu|l4|gb10|gh200> [test IDs/globs]');
  const requester = context.payload.comment.user.login;
  const {data: permission} = await github.rest.repos.getCollaboratorPermissionLevel({owner, repo, username: requester});
  if (!ALLOWED_PERMISSIONS.has(permission.permission)) {
    throw new Error('Only maintainers may authorize lab execution of a PR');
  }
  const {data: pr} = await github.rest.pulls.get({owner, repo, pull_number: context.payload.issue.number});
  if (pr.state !== 'open' || !pr.head.repo) throw new Error('PR must be open with an accessible source repository');

  const marker = `uccl-ci-request-${context.runId}-${attempt}`;
  const header = [
    `<!-- ${marker} -->`,
    `### \`${command}\``,
    '',
    `Commit: \`${pr.head.sha}\` from \`${pr.head.repo.full_name}\``,
    `Requested by: @${requester} ([request](${context.payload.comment.html_url}))`,
    STATUS_MARKER.trim(),
  ].join('\n');
  const dispatcherUrl = runUrl(context, attempt);
  const {data: reply} = await github.rest.issues.createComment({
    owner, repo, issue_number: pr.number,
    body: bodyWithStatus(header, 'queued', dispatcherUrl),
  });
  try {
    await github.rest.actions.createWorkflowDispatch({
      owner, repo, workflow_id: 'ci.yml', ref: context.payload.repository.default_branch,
      inputs: {
        targets: match[2] === 'all' ? '' : match[2], tests: match[3] || '',
        tiers: match[1] === 'run-benchmark' ? 'performance' : 'cpu,build,integration',
        sha: pr.head.sha, head_repo: pr.head.repo.full_name,
        pr: String(pr.number), reply_comment_id: String(reply.id), reply_marker: marker,
      },
    });
  } catch (error) {
    try {
      await github.rest.issues.updateComment({
        owner, repo, comment_id: reply.id,
        body: bodyWithStatus(header, 'failed to dispatch', dispatcherUrl),
      });
    } catch (commentError) {
      core.warning(`Unable to mark the CI comment as failed: ${commentError.message}`);
    }
    throw error;
  }
  core.summary.addRaw(`Dispatched ${match[1]} for PR #${pr.number}, exact SHA ${pr.head.sha}. [Status comment](${reply.html_url}).`);
  await core.summary.write();
}

async function update({github, context, status, results, attempt = process.env.GITHUB_RUN_ATTEMPT || '1'}) {
  if (context.eventName !== 'workflow_dispatch') throw new Error('Comment updates require a dispatched run');
  const inputs = context.payload.inputs;
  if (!/^\d+$/.test(inputs.pr || '') || !/^\d+$/.test(inputs.reply_comment_id || '') ||
      !/^uccl-ci-request-\d+-\d+$/.test(inputs.reply_marker || '')) {
    throw new Error('Invalid CI reply identifiers');
  }
  const {owner, repo} = context.repo;
  const comment_id = Number(inputs.reply_comment_id);
  const {data: reply} = await github.rest.issues.getComment({owner, repo, comment_id});
  const issueUrl = `${context.apiUrl || 'https://api.github.com'}/repos/${owner}/${repo}/issues/${inputs.pr}`;
  const parts = (reply.body || '').split(STATUS_MARKER);
  const commit = `Commit: \`${inputs.sha}\` from \`${inputs.head_repo}\``;
  if (reply.user?.login !== 'github-actions[bot]' || reply.user?.type !== 'Bot' ||
      reply.issue_url !== issueUrl || parts.length !== 2 ||
      !parts[0].startsWith(`<!-- ${inputs.reply_marker} -->\n`) ||
      !parts[0].split('\n').includes(commit)) {
    throw new Error('CI reply does not match this request, PR and source commit');
  }
  if (status !== 'running' && status !== 'complete') throw new Error('Invalid CI comment status');
  const label = status === 'complete' ? completionStatus(results) : 'running';
  await github.rest.issues.updateComment({
    owner, repo, comment_id,
    body: bodyWithStatus(parts[0] + STATUS_MARKER.trimEnd(), label, runUrl(context, attempt), status === 'complete' ? results : null),
  });
}

module.exports = {dispatch, update, completionStatus};
