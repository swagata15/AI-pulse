/**
 * AI Pulse — Frontend Application
 */

const CATEGORY_ICONS = {
    repos: '&#x1F4E6;',        // package
    models: '&#x1F9E0;',       // brain
    news: '&#x1F4F0;',         // newspaper
    papers: '&#x1F4DD;',       // memo
    tools: '&#x1F6E0;',        // wrench
    community: '&#x1F4AC;',    // speech bubble
    influencers: '&#x1F31F;',  // star
};

const CATEGORY_COLORS = {
    repos: '#6366f1',
    models: '#a855f7',
    news: '#f59e0b',
    papers: '#06b6d4',
    tools: '#10b981',
    community: '#ec4899',
    influencers: '#f97316',
};

const MODEL_TYPE_ICONS = {
    chat: '&#x1F4AC;',
    code: '&#x1F4BB;',
    image: '&#x1F3A8;',
    video: '&#x1F3AC;',
    audio: '&#x1F3B5;',
};

let currentCategory = 'all';
let feedData = null;

// ---- Fetch data ----
async function fetchFeed() {
    try {
        const resp = await fetch('/api/feed');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        feedData = await resp.json();
        renderTabs();
        renderFeed();
        updateLastRefresh();
    } catch (err) {
        console.error('Failed to fetch feed:', err);
        showError();
    }
}

// ---- Update last refresh time ----
function updateLastRefresh() {
    const el = document.getElementById('lastUpdated');
    if (feedData?.last_refresh) {
        const dt = new Date(feedData.last_refresh);
        const ago = timeSince(dt);
        el.textContent = `Updated ${ago}`;
    } else {
        el.textContent = 'Fetching data...';
    }
}

function timeSince(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

// ---- Render category tabs ----
function renderTabs() {
    const nav = document.getElementById('categoryNav').querySelector('.nav-inner');
    nav.innerHTML = '';

    // "All" tab
    const allTab = createTab('all', '&#x1F30D;', 'All', getTotalCount());
    nav.appendChild(allTab);

    // Category tabs
    if (feedData?.categories) {
        for (const [key, label] of Object.entries(feedData.categories)) {
            const count = (feedData.items[key] || []).length;
            if (count === 0 && key === 'influencers') continue; // hide if empty
            const icon = CATEGORY_ICONS[key] || '&#x1F4CB;';
            const tab = createTab(key, icon, label, count);
            nav.appendChild(tab);
        }
    }
}

function createTab(category, icon, label, count) {
    const btn = document.createElement('button');
    btn.className = `cat-tab${currentCategory === category ? ' active' : ''}`;
    btn.dataset.category = category;
    btn.innerHTML = `
        <span class="tab-icon">${icon}</span>
        ${label}
        <span class="tab-count">${count}</span>
    `;
    btn.addEventListener('click', () => {
        currentCategory = category;
        document.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        renderFeed();
        // Scroll to top of main content
        document.getElementById('mainContent').scrollIntoView({ behavior: 'smooth' });
    });
    return btn;
}

function getTotalCount() {
    if (!feedData?.items) return 0;
    return Object.values(feedData.items).reduce((sum, arr) => sum + arr.length, 0);
}

// ---- Render TLDR section ----
function renderTldr() {
    const tldr = feedData?.tldr;
    if (!tldr || tldr.length === 0) return '';

    const items = tldr.map((item, idx) => {
        const catColor = CATEGORY_COLORS[item.category] || '#6366f1';
        const catLabel = feedData.categories[item.category] || item.category;
        const catIcon = CATEGORY_ICONS[item.category] || '&#x1F4CB;';
        const score = Math.round(item.normalized_score || 0);

        return `
            <a href="${escapeAttr(item.url)}" target="_blank" class="tldr-item" style="--accent: ${catColor}">
                <div class="tldr-rank">${idx + 1}</div>
                <div class="tldr-content">
                    <div class="tldr-category">
                        <span>${catIcon}</span> ${escapeHtml(catLabel)}
                        <span class="tldr-source">${escapeHtml(item.source)}</span>
                    </div>
                    <h3 class="tldr-title">${escapeHtml(item.title)}</h3>
                    ${item.description ? `<p class="tldr-desc">${cleanHtml(item.description).slice(0, 120)}${cleanHtml(item.description).length > 120 ? '...' : ''}</p>` : ''}
                </div>
                <div class="tldr-score">
                    <span class="score-value">${score}</span>
                    <span class="score-label">score</span>
                </div>
            </a>
        `;
    }).join('');

    return `
        <section class="tldr-section">
            <div class="tldr-header">
                <span class="tldr-icon">&#x26A1;</span>
                <div>
                    <h2 class="tldr-heading">TLDR — Today's Top 5</h2>
                    <p class="tldr-subtitle">The biggest things in AI right now, ranked by engagement across all sources</p>
                </div>
            </div>
            <div class="tldr-list">
                ${items}
            </div>
        </section>
    `;
}

// ---- Render feed ----
function renderFeed() {
    const main = document.getElementById('mainContent');
    const loading = document.getElementById('loadingState');

    if (!feedData?.items) {
        if (loading) loading.style.display = 'flex';
        return;
    }

    if (loading) loading.style.display = 'none';

    let html = '';

    // Show TLDR only on "All" view
    if (currentCategory === 'all') {
        html += renderTldr();
    }

    const categories = currentCategory === 'all'
        ? Object.keys(feedData.categories)
        : [currentCategory];

    let hasContent = false;

    for (const cat of categories) {
        const items = feedData.items[cat] || [];
        if (items.length === 0) continue;
        hasContent = true;

        const catLabel = feedData.categories[cat] || cat;
        const icon = CATEGORY_ICONS[cat] || '&#x1F4CB;';

        if (cat === 'influencers') {
            html += renderInfluencerSection(items, catLabel, icon);
        } else {
            html += `
                <section class="category-section" id="section-${cat}">
                    <div class="section-header">
                        <span class="section-icon">${icon}</span>
                        <h2 class="section-title">${catLabel}</h2>
                        <span class="section-count">${items.length} items</span>
                    </div>
                    <div class="card-grid">
                        ${items.map(item => renderCard(item, cat)).join('')}
                    </div>
                </section>
            `;
        }
    }

    if (!hasContent && currentCategory !== 'all') {
        html += `
            <div class="empty-state">
                <span style="font-size: 48px;">&#x1F50D;</span>
                <p>No items yet. Data is being fetched...</p>
                <p style="font-size: 13px; margin-top: 4px;">Check back in a minute!</p>
            </div>
        `;
    }

    // Keep loading state element but add content
    main.innerHTML = `<div id="loadingState" class="loading-state" style="display:none">
        <div class="spinner"></div>
        <p>Fetching the latest in AI...</p>
    </div>` + html;
}

// ---- Render influencer section ----
function renderInfluencerSection(items, label, icon) {
    const cards = items.map(item => {
        const extra = item.extra || {};
        const topics = extra.topics || [];
        const handle = extra.handle || '';
        const platform = extra.platform || item.source || '';

        return `
            <a href="${escapeAttr(item.url)}" target="_blank" class="influencer-card">
                <div class="influencer-avatar">${escapeHtml(item.title).charAt(0).toUpperCase()}</div>
                <div class="influencer-info">
                    <h3 class="influencer-name">${escapeHtml(item.title)}</h3>
                    ${handle ? `<span class="influencer-handle">${escapeHtml(handle)}</span>` : ''}
                    <span class="influencer-platform">${escapeHtml(platform)}</span>
                </div>
                ${item.description ? `<p class="influencer-desc">${escapeHtml(item.description)}</p>` : ''}
                ${topics.length ? `<div class="card-tags">${topics.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
            </a>
        `;
    }).join('');

    return `
        <section class="category-section" id="section-influencers">
            <div class="section-header">
                <span class="section-icon">${icon}</span>
                <h2 class="section-title">${label}</h2>
                <span class="section-count">${items.length} people</span>
            </div>
            <div class="influencer-grid">
                ${cards}
            </div>
        </section>
    `;
}

// ---- Render a single card ----
function renderCard(item, category) {
    const extra = item.extra || {};
    const score = item.normalized_score || 0;
    const hypeLevel = getHypeLevel(score);
    const hypeHtml = hypeLevel
        ? `<span class="card-hype ${hypeLevel.class}">${hypeLevel.icon} ${hypeLevel.label}</span>`
        : '';

    let metaHtml = '';
    let tagsHtml = '';

    switch (category) {
        case 'repos':
            metaHtml = buildMeta([
                extra.stars != null ? { icon: '&#x2B50;', text: formatNumber(extra.stars) } : null,
                extra.forks ? { icon: '&#x1F500;', text: formatNumber(extra.forks) } : null,
                extra.language ? { icon: '&#x1F4BB;', text: extra.language } : null,
            ]);
            if (extra.topics?.length) {
                tagsHtml = `<div class="card-tags">${extra.topics.map(t => `<span class="tag">${t}</span>`).join('')}</div>`;
            }
            break;

        case 'models': {
            const modelType = extra.model_type || 'chat';
            metaHtml = buildMeta([
                extra.downloads != null ? { icon: '&#x2B07;', text: formatNumber(extra.downloads) + ' downloads' } : null,
                extra.likes != null ? { icon: '&#x2764;', text: formatNumber(extra.likes) } : null,
            ]);
            tagsHtml = `<div class="card-tags">
                <span class="model-type-badge model-type-${modelType}">
                    ${MODEL_TYPE_ICONS[modelType] || ''} ${modelType}
                </span>
                ${extra.pipeline ? `<span class="tag tag-model">${extra.pipeline}</span>` : ''}
            </div>`;
            break;
        }

        case 'community':
            metaHtml = buildMeta([
                extra.ups != null ? { icon: '&#x2B06;', text: formatNumber(extra.ups) } : null,
                extra.points != null ? { icon: '&#x2B06;', text: formatNumber(extra.points) + ' pts' } : null,
                extra.comments != null ? { icon: '&#x1F4AC;', text: formatNumber(extra.comments) } : null,
            ]);
            break;

        case 'papers':
            metaHtml = buildMeta([
                extra.area ? { icon: '&#x1F3F7;', text: extra.area } : null,
                extra.authors ? { icon: '&#x270D;', text: truncate(extra.authors, 40) } : null,
            ]);
            break;

        case 'tools':
            metaHtml = buildMeta([
                extra.stars != null ? { icon: '&#x2B50;', text: formatNumber(extra.stars) } : null,
                extra.language ? { icon: '&#x1F4BB;', text: extra.language } : null,
                extra.owner ? { icon: '&#x1F464;', text: extra.owner } : null,
            ]);
            break;

        case 'news':
            metaHtml = buildMeta([
                extra.published ? { icon: '&#x1F4C5;', text: formatDate(extra.published) } : null,
            ]);
            break;
    }

    const desc = cleanHtml(item.description || '');

    return `
        <article class="card" onclick="window.open('${escapeAttr(item.url)}', '_blank')">
            <div class="card-top">
                <span class="card-source">${escapeHtml(item.source)}</span>
                ${hypeHtml}
            </div>
            <h3 class="card-title">${escapeHtml(item.title)}</h3>
            ${desc ? `<p class="card-description">${desc}</p>` : ''}
            <div class="card-meta">${metaHtml}</div>
            ${tagsHtml}
        </article>
    `;
}

// ---- Hype level (now based on normalized 0-100 score) ----
function getHypeLevel(score) {
    if (score >= 60) return { class: 'hype-hot', icon: '&#x1F525;', label: 'HOT' };
    if (score >= 30) return { class: 'hype-trending', icon: '&#x1F4C8;', label: 'Trending' };
    if (score >= 10) return { class: 'hype-new', icon: '&#x2728;', label: 'New' };
    return null;
}

// ---- Build meta items ----
function buildMeta(items) {
    return items
        .filter(Boolean)
        .map(m => `<span class="meta-item"><span class="meta-icon">${m.icon}</span> ${escapeHtml(String(m.text))}</span>`)
        .join('');
}

// ---- Helpers ----
function formatNumber(n) {
    if (n == null) return '0';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    try {
        const d = new Date(dateStr);
        if (isNaN(d)) return dateStr;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch {
        return dateStr;
    }
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function cleanHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.innerHTML = str;
    return div.textContent || div.innerText || '';
}

function showError() {
    const main = document.getElementById('mainContent');
    main.innerHTML = `
        <div class="empty-state">
            <span style="font-size: 48px;">&#x26A0;</span>
            <p>Failed to load feed. Please try refreshing.</p>
        </div>
    `;
}

// ---- Refresh button ----
document.getElementById('refreshBtn').addEventListener('click', async () => {
    const btn = document.getElementById('refreshBtn');
    btn.classList.add('spinning');
    btn.disabled = true;

    try {
        await fetch('/api/refresh', { method: 'POST' });
        document.getElementById('lastUpdated').textContent = 'Refreshing...';

        setTimeout(async () => {
            await fetchFeed();
            btn.classList.remove('spinning');
            btn.disabled = false;
        }, 5000);
    } catch (err) {
        console.error('Refresh failed:', err);
        btn.classList.remove('spinning');
        btn.disabled = false;
    }
});

// ---- Auto-refresh every 30 minutes ----
setInterval(fetchFeed, 30 * 60 * 1000);

// ---- Poll more frequently at startup to catch initial data load ----
let startupPolls = 0;
const startupPollInterval = setInterval(() => {
    startupPolls++;
    fetchFeed();
    if (startupPolls >= 12 || (feedData && getTotalCount() > 0)) {
        clearInterval(startupPollInterval);
    }
}, 10000);

// ---- Initial load ----
fetchFeed();
