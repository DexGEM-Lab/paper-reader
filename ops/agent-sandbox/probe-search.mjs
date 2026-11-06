// Operator smoke test: invoke the ACTUAL pinned plugin, without an agent/model
// conversation. Uses only the container's separately provisioned limited key.
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
const require = createRequire('/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/package.json');
const { createJiti } = require('jiti');
const load = createJiti(import.meta.url, {
  alias: {
    '@earendil-works/pi-coding-agent': '/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/dist/index.js',
    '@earendil-works/pi-tui': '/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-tui/dist/index.js',
  },
});
const plugin = await load.import('/opt/pi-native-search/extensions/index.ts', { default: true });
const tools = {};
plugin({ registerTool: tool => { tools[tool.name] = tool; }, registerCommand() {}, on() {} });
const ctx = {
  model: {
    provider: 'dexgem-messages', id: 'claude-sonnet-5-5', api: 'anthropic-messages',
    baseUrl: process.env.PAPER_READER_SEARCH_BASE_URL || 'https://litellm.dex-gem.ai',
  },
  modelRegistry: { getApiKeyForProvider: async () => process.env.PAPER_READER_LLM_API_KEY },
};
const result = await tools.web_search.execute('probe', {
  query: 'Find the original Attention Is All You Need paper. Include its arxiv URL 1706.03762.',
}, AbortSignal.timeout(90000), undefined, ctx);
writeFileSync(process.argv[2] || 'search-probe.json', JSON.stringify(result, null, 2));
if (result.isError || result.details?.method !== 'native' || !JSON.stringify(result).includes('arxiv.org')) {
  console.error('Search probe failed; inspect search-probe.json (contains no credential).');
  process.exitCode = 1;
} else {
  console.log('Native web_search returned arxiv.org sources from inside the shared sandbox.');
}
