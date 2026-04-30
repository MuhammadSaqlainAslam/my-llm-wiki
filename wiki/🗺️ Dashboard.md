---
title: "Dashboard"
tags: [meta, dashboard]
---

# 🗺️ Dashboard

Live Dataview queries over the `wiki/` vault. Requires the **Dataview** community plugin (enable JavaScript queries in settings).

> **GitHub:** [MuhammadSaqlainAslam/my-llm-wiki](https://github.com/MuhammadSaqlainAslam/my-llm-wiki)

---

## 🆕 Latest Addition

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  join(tags, " · ") AS "Tags",
  tldr AS "TL;DR"
FROM "wiki"
WHERE title != null
  AND file.name != "Dashboard"
  AND file.name != "Home"
  AND file.name != "000 Index"
SORT file.mtime DESC
LIMIT 1
```

---

## 📚 All Notes

All notes with frontmatter, sorted by year then title. Notes without a `year` field appear at the bottom.

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  join(tags, " · ") AS "Tags",
  tldr AS "TL;DR"
FROM "wiki"
WHERE title != null
  AND file.name != "Dashboard"
  AND file.name != "Home"
  AND file.name != "000 Index"
SORT year ASC, title ASC
```

---

## 🗂️ By Theme

### 🔷 Foundations
*Self-attention and the KV cache — what everything else is reacting to.*

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  tldr AS "TL;DR"
FROM "wiki"
WHERE theme = "foundations"
SORT year ASC
```

---

### ⚡ Efficiency
*Replacing or compressing O(n²) attention — SSMs, kernel fusion, compressed sparse attention, KV cache reduction.*

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  tldr AS "TL;DR"
FROM "wiki"
WHERE theme = "efficiency"
SORT year ASC
```

---

### 📈 Scaling
*Decoupling parameter count from per-token compute — MoE routing and its variants.*

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  tldr AS "TL;DR"
FROM "wiki"
WHERE theme = "scaling"
SORT year ASC
```

---

### 🔬 Synthesis
*Production systems that combine all three themes into a single architecture.*

```dataview
TABLE
  title AS "Title",
  year AS "Year",
  tldr AS "TL;DR"
FROM "wiki"
WHERE theme = "synthesis"
SORT year ASC
```

---

## 🔗 Orphan Links

Pages that are `[[linked]]` somewhere in the vault but have no corresponding `.md` file yet. Uses `app.metadataCache.unresolvedLinks` — the same data Obsidian uses to colour broken links red in the graph view.

```dataviewjs
const folder = "wiki";
const unresolvedLinks = app.metadataCache.unresolvedLinks;

// Collect all unresolved links from files inside wiki/
const orphans = new Map(); // linkName -> Set of source file names

for (const [sourcePath, links] of Object.entries(unresolvedLinks)) {
    if (!sourcePath.startsWith(folder + "/")) continue;
    for (const [linkName, count] of Object.entries(links)) {
        if (linkName.trim() === "") continue;
        if (!orphans.has(linkName)) orphans.set(linkName, new Set());
        const shortName = sourcePath
            .replace(folder + "/", "")
            .replace(/\.md$/, "");
        orphans.get(linkName).add(shortName);
    }
}

if (orphans.size === 0) {
    dv.paragraph("✅ **No orphan links.** Every `[[wikilink]]` resolves to an existing note.");
} else {
    const rows = [...orphans.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([name, sources]) => [
            `\`[[${name}]]\``,
            [...sources].map(s => `[[${s}]]`).join(", "),
            sources.size
        ]);
    dv.table(
        ["Missing Page", "Linked From", "# References"],
        rows
    );
    dv.paragraph(`⚠️ **${orphans.size} orphan link(s) found.** Create a note for each to resolve them.`);
}
```

---

## 🕒 Recently Modified

The 5 most recently edited notes — useful for picking up where you left off.

```dataview
TABLE
  title AS "Title",
  dateformat(file.mtime, "yyyy-MM-dd HH:mm") AS "Last Modified",
  theme AS "Theme",
  tldr AS "TL;DR"
FROM "wiki"
WHERE title != null
  AND file.name != "Dashboard"
SORT file.mtime DESC
LIMIT 5
```

---

## 📊 Vault Stats

```dataviewjs
const pages = dv.pages('"wiki"')
    .where(p => p.title != null && p.file.name !== "Dashboard");

const total = pages.length;
const byTheme = {};
for (const p of pages) {
    const t = p.theme ?? "unthemed";
    byTheme[t] = (byTheme[t] ?? 0) + 1;
}

const withYear   = pages.filter(p => p.year  != null).length;
const withTldr   = pages.filter(p => p.tldr  != null).length;
const withTheme  = pages.filter(p => p.theme != null).length;

dv.paragraph(
    `**${total}** notes · ` +
    `**${withTheme}** themed · ` +
    `**${withYear}** with year · ` +
    `**${withTldr}** with TL;DR`
);

dv.table(
    ["Theme", "Count"],
    Object.entries(byTheme).sort(([,a],[,b]) => b - a)
);
```
