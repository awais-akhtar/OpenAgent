import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { build } from 'vite';

const root = process.cwd();
const output = path.resolve(root, 'dist');
const productionWeb = path.resolve(root, '../web');
if (output === productionWeb || output.startsWith(`${productionWeb}${path.sep}`) || productionWeb.startsWith(`${output}${path.sep}`)) {
  throw new Error(`Unsafe build output: ${output}`);
}
function digestTree(directory) {
  if (!existsSync(directory)) return 'absent';
  const records = [];
  const visit = (current) => {
    for (const entry of readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) visit(full);
      else if (entry.isFile()) records.push(`${path.relative(directory, full)}:${createHash('sha256').update(readFileSync(full)).digest('hex')}`);
    }
  };
  visit(directory);
  return createHash('sha256').update(records.join('\n')).digest('hex');
}
const webBefore = digestTree(productionWeb);
await build();
if (!existsSync(path.join(output, 'index.html'))) throw new Error('Safe build did not create dist/index.html.');
const artifacts = [];
const visit = (directory) => {
  for (const name of readdirSync(directory)) {
    const full = path.join(directory, name);
    if (statSync(full).isDirectory()) visit(full);
    else artifacts.push(full);
  }
};
visit(output);
const forbiddenNames = artifacts.filter(file => /(?:reference-fixtures|approved-shell-fixtures)\.json$/i.test(file) || path.relative(output, file).split(path.sep).includes('demo'));
if (forbiddenNames.length) throw new Error(`Demo fixtures were copied into the production build: ${forbiddenNames.join(', ')}`);
for (const file of artifacts) {
  const bytes = readFileSync(file);
  for (const fixtureName of ['reference-fixtures.json', 'approved-shell-fixtures.json']) {
    const fixture = readFileSync(path.join(root, 'public', 'demo', fixtureName));
    if (bytes.includes(fixture)) throw new Error(`Demo fixture data leaked into build artifact: ${file}`);
  }
}
if (digestTree(productionWeb) !== webBefore) throw new Error('Safe frontend build modified ../web; restore is not automatic, investigate before continuing.');
console.log(`Safe Vite build verified at ${output}. Python-served ../web is unchanged; public/demo fixtures are absent.`);
