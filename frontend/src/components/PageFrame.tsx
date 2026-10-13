import type { PropsWithChildren, ReactNode } from 'react';

export function PageFrame({ eyebrow, title, description, actions, children }: PropsWithChildren<{ eyebrow: string; title: string; description: string; actions?: ReactNode }>) {
  return <main className="page"><header className="page-head"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="lede">{description}</p></div>{actions && <div className="head-actions">{actions}</div>}</header>{children}</main>;
}
