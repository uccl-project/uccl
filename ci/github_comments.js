// Loaded only from the trusted workflow commit by jobs with comment permission.
const {suiteResult} = require('./github_results.js');
const STATUS_MARKER = '\n<!-- uccl-ci-status -->\n';
const ALLOWED_PERMISSIONS = new Set(['write', 'maintain', 'admin']);
const USAGE = 'Usage: /run-ci|/run-benchmark <all|cpu|l4|gb10|gh200> [test IDs/globs]';
const STATUS_LABELS = {
  queued: '⏳ Queued',
  running: '🔄 Running',
  'completed successfully': '✅ Pass',
  failed: '❌ Failed',
  'failed to dispatch': '⚠️ Failed to dispatch',
  cancelled: '⚠️ Cancelled',
  skipped: '— Skipped',
  incomplete: '⚠️ Incomplete',
  unknown: '❔ Unknown or unclassified',
};
const RESULT_LABELS = {
  success: '✅ Pass',
  failure: '❌ Failed',
  cancelled: '⚠️ Cancelled',
  skipped: '— Skipped',
  incomplete: '⚠️ Incomplete',
  unknown: '❔ Unknown or unclassified',
};

function runUrl(context, attempt) {
  const {owner, repo} = context.repo;
  return `${context.serverUrl || 'https://github.com'}/${owner}/${repo}/actions/runs/${context.runId}/attempts/${attempt}`;
}

function jobResult(results, id, suites) {
  const state = results[id].result;
  return state === 'success' && ['cpu', 'gpu'].includes(id) ? suiteResult(suites[id]) : state;
}

function briefBodyWithStatus(header, requester, status, url) {
  const [marker, ...details] = header.split('\n');
  // Keep request validation metadata out of the short visible acknowledgement.
  const lines = [marker, '<!--', ...details, '-->', STATUS_MARKER.trim(),
    `@${requester} **Status: ${STATUS_LABELS[status]}** · [Workflow run and logs](${url})`];
  if (status === 'queued' || status === 'running') {
    lines.push('', 'I’ll post the full results when the run finishes.');
  }
  return lines.join('\n');
}

function bodyWithStatus(header, status, url, results, suites = {}) {
  const lines = [header, `**Status: ${STATUS_LABELS[status] || '❔ Unknown or unclassified'}**`, '', `[Workflow run and logs](${url})`];
  if (results) {
    lines.push('', '| Job | Result |', '|---|---|');
    for (const [id, label] of [['plan', 'Plan'], ['cpu', 'CPU'], ['gpu', 'GPU'], ['status', 'CI status']]) {
      const counts = ['cpu', 'gpu'].includes(id) && results[id].result !== 'skipped' ? suites[id] : null;
      const detail = counts ? ` (${counts.executed} executed, ${counts.skipped} skipped${counts.missing ? `, ${counts.missing} missing` : ''})` : '';
      lines.push(`| ${label} | ${RESULT_LABELS[jobResult(results, id, suites)] || '❔ Unknown or unclassified'}${detail} |`);
    }
    lines.push('', 'CPU/GPU counts are suites per target, not ranks. Missing or unreadable results are unknown; CI status is the workflow gate.');
    lines.push('', 'Per-suite results, skips and timings are in the run summary and artifacts.');
    if (['failure', 'cancelled'].includes(results['comment-start']?.result)) {
      lines.push('', '⚠️ Note: the initial running-status update did not succeed. The CI result above reflects the test jobs, not that notification failure.');
    }
  }
  return lines.join('\n');
}

function completionStatus(results, suites = {}) {
  const states = ['plan', 'cpu', 'gpu', 'status'].map(id => results[id]?.result);
  if (states.some(state => !['success', 'failure', 'cancelled', 'skipped'].includes(state))) {
    throw new Error('Missing or invalid CI job results');
  }
  if (states.includes('cancelled')) return 'cancelled';
  if (states.includes('failure')) return 'failed';
  const tests = ['cpu', 'gpu'].map(id => jobResult(results, id, suites));
  if (tests.includes('failure')) return 'failed';
  if (tests.includes('unknown')) return 'unknown';
  if (tests.every(state => state === 'skipped')) return 'skipped';
  if (tests.includes('incomplete') || ['cpu', 'gpu'].some(id => results[id].result === 'success' && suiteResult(suites[id]) === 'skipped')) return 'incomplete';
  return 'completed successfully';
}

async function dispatch({github, context, core, attempt = process.env.GITHUB_RUN_ATTEMPT || '1'}) {
  const {owner, repo} = context.repo;
  if (!context.payload.issue?.pull_request) throw new Error('CI commands require a pull request');
  const command = context.payload.comment.body.trim();
  const requester = context.payload.comment.user.login;
  const {data: permission} = await github.rest.repos.getCollaboratorPermissionLevel({owner, repo, username: requester});
  if (!ALLOWED_PERMISSIONS.has(permission.permission)) {
    throw new Error('Only maintainers may authorize lab execution of a PR');
  }
  const match = command.match(/^\/(run-ci|run-benchmark)\s+(all|cpu|l4|gb10|gh200)(?:\s+([a-z0-9_.*,-]+))?$/);
  if (!match) {
    await github.rest.issues.createComment({
      owner, repo, issue_number: context.payload.issue.number,
      body: `@${requester} I could not parse that CI command.\n\n\`${USAGE}\`\n\nUse test IDs/globs such as \`p2p.*\`, not file paths. For example: \`/run-ci l4 p2p.*\` or \`/run-benchmark gh200 ep.bench.*\`.`,
    });
    throw new Error(USAGE);
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
  ].join('\n');
  const dispatcherUrl = runUrl(context, attempt);
  const {data: reply} = await github.rest.issues.createComment({
    owner, repo, issue_number: pr.number,
    body: briefBodyWithStatus(header, requester, 'queued', dispatcherUrl),
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
        body: briefBodyWithStatus(header, requester, 'failed to dispatch', dispatcherUrl),
      });
    } catch (commentError) {
      core.warning(`Unable to mark the CI comment as failed: ${commentError.message}`);
    }
    throw error;
  }
  core.summary.addRaw(`Dispatched ${match[1]} for PR #${pr.number}, exact SHA ${pr.head.sha}. [Status comment](${reply.html_url}).`);
  await core.summary.write();
}

async function update({github, context, status, results, suites = {}, attempt = process.env.GITHUB_RUN_ATTEMPT || '1'}) {
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
  // Also accept acknowledgements created before request details were hidden.
  const headerLines = parts[0].split('\n');
  if (headerLines[1] === '<!--' && headerLines.at(-1) === '-->') {
    headerLines.splice(1, 1);
    headerLines.pop();
  }
  const header = headerLines.join('\n');
  const requester = header.match(/^Requested by: @([a-z0-9][a-z0-9-]{0,38}(?:\[bot\])?) \(\[request\]\(/im)?.[1];
  if (!requester) throw new Error('CI reply has no valid requester');
  const label = status === 'complete' ? completionStatus(results, suites) : 'running';
  const url = runUrl(context, attempt);
  await github.rest.issues.updateComment({
    owner, repo, comment_id, body: briefBodyWithStatus(header, requester, label, url),
  });
  if (status === 'complete') {
    // The new full report notifies the requester. Retries update that report
    // without posting another copy, including after an accepted but lost POST.
    const marker = `<!-- uccl-ci-complete-${context.runId}-${attempt} -->`;
    const body = `${marker}\n${bodyWithStatus(header + STATUS_MARKER.trimEnd(), label, url, results, suites)}`;
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner, repo, issue_number: Number(inputs.pr), per_page: 100,
    });
    const previous = comments.find(comment => comment.user?.login === 'github-actions[bot]' &&
      comment.user?.type === 'Bot' && comment.body?.startsWith(`${marker}\n`));
    if (!previous) {
      await github.rest.issues.createComment({
        owner, repo, issue_number: Number(inputs.pr), body,
      });
    } else if (previous.body !== body) {
      await github.rest.issues.updateComment({owner, repo, comment_id: previous.id, body});
    }
  }
}

module.exports = {dispatch, update, completionStatus};
