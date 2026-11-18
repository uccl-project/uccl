// Read PR-produced artifacts as bounded JSON data, never as executable code.
const fs = require('node:fs');
const path = require('node:path');
const ID = /^[a-z0-9][a-z0-9_.-]{0,80}$/;
const TARGET = /^[a-z0-9][a-z0-9_-]{0,63}$/;

function readJson(root, ...parts) {
  try {
    let file = root;
    for (const part of parts) {
      file = path.join(file, part);
      if (fs.lstatSync(file).isSymbolicLink()) return null;
    }
    const stat = fs.statSync(file);
    if (!stat.isFile() || stat.size > 16 * 1024 * 1024) return null;
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

function readSuiteResults(root) {
  const plan = readJson(root, 'ci-plan', 'ci-plan.json');
  if (!Array.isArray(plan?.include)) return {};
  const targets = new Set();
  for (const item of plan.include) {
    if (!item || typeof item.target !== 'string' || !TARGET.test(item.target) ||
        targets.has(item.target) || !Array.isArray(item.tests) ||
        item.tests.some(id => typeof id !== 'string' || !ID.test(id))) return {};
    targets.add(item.target);
  }
  const summaries = {};
  for (const {target, tests} of plan.include) {
    const job = target === 'cpu' ? 'cpu' : 'gpu';
    const counts = summaries[job] ||= {executed: 0, skipped: 0, failed: 0, missing: 0, infrastructure: false};
    const data = readJson(root, job === 'cpu' ? 'ci-cpu' : `ci-gpu-${target}`, target, 'results.json');
    const byId = new Map();
    if (Array.isArray(data?.results)) {
      for (const record of data.results) {
        if (!record || typeof record.id !== 'string' || !ID.test(record.id)) continue;
        if (!byId.has(record.id)) byId.set(record.id, []);
        byId.get(record.id).push(record);
      }
    }
    counts.infrastructure ||= Boolean(data?.infrastructure_error);
    for (const id of new Set(tests)) {
      const ranks = (byId.get(id) || []).filter(r => !r.not_participating);
      if (!ranks.length || ranks.some(r => !Number.isInteger(r.exit_code))) {
        counts.missing++;
      } else if (ranks.some(r => !r.skip_reason && r.exit_code !== 0)) {
        counts.executed++;
        counts.failed++;
      } else if (ranks.some(r => r.skip_reason)) {
        counts.skipped++;
      } else {
        counts.executed++;
      }
    }
  }
  return summaries;
}

function suiteResult(counts) {
  if (!counts) return 'unknown';
  if (counts.failed || counts.infrastructure) return 'failure';
  if (counts.missing || counts.executed + counts.skipped === 0) return 'unknown';
  if (!counts.executed) return 'skipped';
  return counts.skipped ? 'incomplete' : 'success';
}

module.exports = {readSuiteResults, suiteResult};
