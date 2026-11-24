export function escapeHtml(str = '') {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInline(text) {
  let out = escapeHtml(text);
  // Inline code
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Italic
  out = out.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>');
  // Clean common LaTeX readability symbols (since zero external KaTeX)
  out = out
    .replace(/\$\$([^$]+)\$\$/g, '<code>$1</code>')
    .replace(/\$([^$\n]+)\$/g, '<code>$1</code>')
    .replace(/\\times/g, '×')
    .replace(/\\approx/g, '≈')
    .replace(/\\le/g, '≤')
    .replace(/\\ge/g, '≥')
    .replace(/\\ll/g, '≪')
    .replace(/\\sim/g, '~')
    .replace(/\\rightarrow/g, '→')
    .replace(/\\leftarrow/g, '←')
    .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
    .replace(/\\text\{([^}]+)\}/g, '$1')
    .replace(/\\mathbf\{([^}]+)\}/g, '<strong>$1</strong>')
    .replace(/\\underbrace\{([^}]+)\}_\{([^}]+)\}/g, '$1 ($2)')
    .replace(/\\mathbb\{R\}/g, 'ℝ')
    .replace(/\\pi/g, 'π')
    .replace(/\\theta/g, 'θ')
    .replace(/\\alpha/g, 'α')
    .replace(/\\beta/g, 'β')
    .replace(/\\gamma/g, 'γ')
    .replace(/\\Delta/g, 'Δ')
    .replace(/\\nabla/g, '∇')
    .replace(/\\Psi/g, 'Ψ')
    .replace(/\\cdot/g, '·')
    .replace(/\\in/g, '∈')
    .replace(/\\min/g, 'min')
    .replace(/\\max/g, 'max')
    .replace(/\\log/g, 'log')
    .replace(/\\left|\\right/g, '');
  return out;
}

export function renderMarkdown(md = '') {
  if (!md) return '';
  const lines = String(md).replace(/\r\n/g, '\n').split('\n');
  const html = [];
  let inCode = false;
  let codeLines = [];
  let inList = null; // 'ul' | 'ol'
  let tableRows = [];

  const flushList = () => {
    if (inList) {
      html.push(`</${inList}>`);
      inList = null;
    }
  };

  const flushTable = () => {
    if (!tableRows.length) return;
    const rows = tableRows.filter((r) => !/^\s*\|?[\s:-]+\|[\s|:-]*$/.test(r));
    if (!rows.length) {
      tableRows = [];
      return;
    }
    const parseCells = (row) =>
      row
        .trim()
        .replace(/^\||\|$/g, '')
        .split('|')
        .map((c) => c.trim());

    const headers = parseCells(rows[0]);
    const body = rows.slice(1).map(parseCells);

    html.push('<div style="overflow-x:auto;margin:8px 0;"><table class="md-table">');
    html.push('<thead><tr>' + headers.map((h) => `<th>${renderInline(h)}</th>`).join('') + '</tr></thead>');
    html.push(
      '<tbody>' +
        body
          .map((cells) => '<tr>' + cells.map((c) => `<td>${renderInline(c)}</td>`).join('') + '</tr>')
          .join('') +
        '</tbody>'
    );
    html.push('</table></div>');
    tableRows = [];
  };

  for (const rawLine of lines) {
    const line = rawLine;
    if (line.trim().startsWith('```')) {
      flushList();
      flushTable();
      if (!inCode) {
        inCode = true;
        codeLines = [];
      } else {
        html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        inCode = false;
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (line.trim().startsWith('|') && line.includes('|')) {
      flushList();
      tableRows.push(line);
      continue;
    } else if (tableRows.length) {
      flushTable();
    }

    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      continue;
    }

    if (/^#{1,4}\s+/.test(trimmed)) {
      flushList();
      const level = trimmed.match(/^(#{1,4})/)[1].length;
      const tag = `h${Math.min(level + 2, 6)}`;
      const content = trimmed.replace(/^#{1,4}\s+/, '');
      html.push(`<${tag} style="margin:12px 0 6px;font-weight:700;">${renderInline(content)}</${tag}>`);
      continue;
    }

    if (trimmed.startsWith('> ')) {
      flushList();
      html.push(
        `<blockquote style="margin:8px 0;padding:8px 12px;border-left:3px solid var(--accent);background:var(--bg-elevated);border-radius:0 6px 6px 0;">${renderInline(trimmed.slice(2))}</blockquote>`
      );
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      if (inList !== 'ul') {
        flushList();
        inList = 'ul';
        html.push('<ul>');
      }
      html.push(`<li>${renderInline(trimmed.replace(/^[-*]\s+/, ''))}</li>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      if (inList !== 'ol') {
        flushList();
        inList = 'ol';
        html.push('<ol>');
      }
      html.push(`<li>${renderInline(trimmed.replace(/^\d+\.\s+/, ''))}</li>`);
      continue;
    }

    flushList();
    html.push(`<p>${renderInline(trimmed)}</p>`);
  }

  flushList();
  flushTable();
  if (inCode) {
    html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  }
  return html.join('');
}
