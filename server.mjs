import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PUBLIC_DIR = path.join(__dirname, 'public');
const PORT = Number(process.env.PORT || 3456);
const HOST = process.env.HOST || '127.0.0.1';

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.md': 'text/markdown; charset=utf-8',
};

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

async function readBody(req, maxBytes = 1024 * 1024) {
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > maxBytes) {
      throw new Error('Request body too large');
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function handleLlmProxy(req, res) {
  try {
    const raw = await readBody(req);
    const body = JSON.parse(raw || '{}');
    const { baseUrl, apiKey, model, messages, temperature = 0.4, max_tokens = 1400 } = body;

    if (!baseUrl || !model || !Array.isArray(messages)) {
      return sendJson(res, 400, {
        error: 'Missing required fields: baseUrl, model, messages',
      });
    }

    let targetUrl;
    try {
      const normalized = baseUrl.replace(/\/+$/, '');
      targetUrl = normalized.endsWith('/chat/completions')
        ? normalized
        : `${normalized}/chat/completions`;
      const parsed = new URL(targetUrl);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return sendJson(res, 400, { error: 'Only http/https endpoints are supported' });
      }
    } catch {
      return sendJson(res, 400, { error: 'Invalid baseUrl format' });
    }

    const headers = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000);

    try {
      const upstream = await fetch(targetUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model,
          messages,
          temperature,
          max_tokens,
        }),
        signal: controller.signal,
      });

      const text = await upstream.text();
      if (!upstream.ok) {
        return sendJson(res, upstream.status, {
          error: `Upstream LLM returned HTTP ${upstream.status}`,
          details: text.slice(0, 800),
        });
      }

      const data = JSON.parse(text);
      const content = data?.choices?.[0]?.message?.content ?? '';
      return sendJson(res, 200, {
        content,
        usage: data?.usage || null,
        model: data?.model || model,
      });
    } finally {
      clearTimeout(timer);
    }
  } catch (err) {
    const msg = err?.name === 'AbortError' ? 'Upstream request timed out (45s)' : err.message;
    return sendJson(res, 502, { error: msg || 'Proxy error' });
  }
}

async function serveStatic(req, res) {
  const urlObj = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  let reqPath = decodeURIComponent(urlObj.pathname);
  if (reqPath === '/') reqPath = '/index.html';

  const safePath = path.normalize(reqPath).replace(/^(\.\.(\/|\\|$))+/, '');
  const fullPath = path.join(PUBLIC_DIR, safePath);

  if (!fullPath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  try {
    const stat = await fs.stat(fullPath);
    const filePath = stat.isDirectory() ? path.join(fullPath, 'index.html') : fullPath;
    const ext = path.extname(filePath).toLowerCase();
    const content = await fs.readFile(filePath);
    res.writeHead(200, {
      'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(content);
  } catch (err) {
    if (err.code === 'ENOENT') {
      // Fallback to index.html for SPA routes without file extension
      if (!path.extname(reqPath)) {
        try {
          const indexContent = await fs.readFile(path.join(PUBLIC_DIR, 'index.html'));
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
          res.end(indexContent);
          return;
        } catch {
          // ignore
        }
      }
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('500 Internal Server Error');
  }
}

const server = http.createServer(async (req, res) => {
  if (req.url === '/api/health' && req.method === 'GET') {
    return sendJson(res, 200, { ok: true, service: 'ai-infra-interviewer', timestamp: Date.now() });
  }
  if (req.url === '/api/llm-proxy' && req.method === 'POST') {
    return handleLlmProxy(req, res);
  }
  if (req.method === 'GET' || req.method === 'HEAD') {
    return serveStatic(req, res);
  }
  res.writeHead(405);
  res.end('Method Not Allowed');
});

server.listen(PORT, HOST, () => {
  console.log(`\n  ⚡ AI Infra Interviewer running at http://${HOST}:${PORT}\n`);
});
