import type { PropsWithChildren, ReactNode } from 'react';

type PageFrameProps = PropsWithChildren<{ eyebrow?: string; title: string; description?: string; actions?: ReactNode }>;

export function PageFrame({ title, actions, children }: PageFrameProps) {
  // Keep the shared frame quiet: specialist pages already communicate context
  // through their controls and content, so repeating category + description
  // above every view only adds noise.
  return <main className="page"><header className="page-head"><h1>{title}</h1>{actions && <div className="head-actions">{actions}</div>}</header>{children}</main>;
}
