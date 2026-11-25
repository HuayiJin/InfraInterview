import { escapeHtml } from '../engine/markdown.js';

export function renderVideoStrip(videos, { heading = 'YouTube 垂类图解视频', compact = false } = {}) {
  if (!Array.isArray(videos) || videos.length === 0) return '';

  return `
    <div class="video-strip-box ${compact ? 'compact' : ''}">
      <div class="video-strip-head">
        <span class="video-strip-title">
          <span class="yt-play-dot" aria-hidden="true">▶</span>
          <span>${escapeHtml(heading)}</span>
        </span>
        <span class="video-strip-sub">精选短视频 / 动画图解</span>
      </div>
      <div class="video-card-list">
        ${videos
          .map(
            (v) => `
          <a
            class="video-link-card"
            href="${escapeHtml(v.url)}"
            target="_blank"
            rel="noopener noreferrer"
            title="在 YouTube 打开：${escapeHtml(v.title)}"
          >
            <span class="video-play-badge" aria-hidden="true">▶</span>
            <span class="video-card-main">
              <span class="video-card-note">${escapeHtml(v.note)}</span>
              <span class="video-card-meta">
                <span class="video-card-orig">${escapeHtml(v.title)}</span>
              </span>
              <span class="video-card-tags">
                <span class="video-tag-channel">${escapeHtml(v.channel)}</span>
                <span class="video-tag-style">${escapeHtml(v.styleTag)}</span>
                <span class="video-tag-lang">${escapeHtml(v.lang)}</span>
              </span>
            </span>
            <span class="video-ext-arrow" aria-hidden="true">↗</span>
          </a>
        `
          )
          .join('')}
      </div>
    </div>
  `;
}
