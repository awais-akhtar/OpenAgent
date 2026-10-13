import { Check, Copy } from 'lucide-react';
import type { ReactNode } from 'react';

function safeHref(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null;
  } catch { return null; }
}

function inline(value: string, keyPrefix: string) {
  const result: ReactNode[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\)|https?:\/\/[^\s<]+)/g;
  let offset = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(value))) {
    if (match.index > offset) result.push(value.slice(offset, match.index));
    const token = match[0];
    if (token.startsWith('`')) result.push(<code key={`${keyPrefix}-${index++}`}>{token.slice(1, -1)}</code>);
    else if (token.startsWith('**')) result.push(<strong key={`${keyPrefix}-${index++}`}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith('*')) result.push(<em key={`${keyPrefix}-${index++}`}>{token.slice(1, -1)}</em>);
    else {
      const markdown = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const label = markdown?.[1] ?? token;
      const href = safeHref(markdown?.[2] ?? token);
      result.push(href ? <a key={`${keyPrefix}-${index++}`} href={href} target="_blank" rel="noopener noreferrer">{label}</a> : token);
    }
    offset = match.index + token.length;
  }
  if (offset < value.length) result.push(value.slice(offset));
  return result;
}

export function MessageContent({ content }: { content: string }) {
  const lines = content.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let code: string[] | null = null;
  let language = '';
  let index = 0;
  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(<p key={`p-${index}`}>{inline(paragraph.join('\n'), `p-${index++}`)}</p>);
    paragraph = [];
  };
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (line.startsWith('```')) {
      if (code) {
        const source = code.join('\n');
        const blockIndex = index++;
        blocks.push(<div className="oa-chat-codeblock" key={`code-${blockIndex}`}><div className="oa-chat-code-head"><span>{language || 'Code'}</span><button type="button" onClick={event => { const button = event.currentTarget; void navigator.clipboard?.writeText(source).then(() => { button.dataset.copied = 'true'; window.setTimeout(() => { delete button.dataset.copied; }, 1400); }); }} aria-label="Copy code"><Copy size={13}/><Check size={13}/><span>Copy</span></button></div><pre><code>{source}</code></pre></div>);
        code = null; language = '';
      } else {
        flushParagraph(); code = []; language = line.slice(3).trim();
      }
    } else if (code) code.push(line);
    else if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      const list: string[] = [];
      while (lineIndex < lines.length && /^\s*[-*]\s+/.test(lines[lineIndex])) {
        list.push(lines[lineIndex].replace(/^\s*[-*]\s+/, ''));
        lineIndex += 1;
      }
      lineIndex -= 1;
      blocks.push(<ul key={`list-${index}`}>{list.map((item, itemIndex) => <li key={itemIndex}>{inline(item, `list-${index}-${itemIndex}`)}</li>)}</ul>);
      index += 1;
    } else if (/^#{1,6}\s+/.test(line)) {
      flushParagraph();
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      blocks.push(<h3 key={`heading-${index++}`}>{inline(heading?.[2] ?? line, `heading-${index}`)}</h3>);
    } else if (!line.trim()) flushParagraph();
    else paragraph.push(line);
  }
  if (code) blocks.push(<pre key={`code-${index++}`}><code>{code.join('\n')}</code></pre>);
  flushParagraph();
  return <div className="oa-chat-markdown">{blocks}</div>;
}
