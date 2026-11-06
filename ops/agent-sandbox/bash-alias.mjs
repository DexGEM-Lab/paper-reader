// Some models invoke the conventional capitalized name despite Pi registering
// its shell tool as lowercase. Delegate to Pi's own implementation so both
// spellings have identical cwd, timeout, cancellation and output semantics.
import { createBashToolDefinition } from '/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/dist/core/tools/bash.js';

export default function registerBashAlias(pi) {
  const bash = createBashToolDefinition(process.cwd());
  pi.registerTool({ ...bash, name: 'Bash', label: 'Bash',
    description: 'Alias for bash. Execute a shell command in the current working directory.' });
}
