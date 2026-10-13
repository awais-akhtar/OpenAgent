import * as TabsPrimitive from '@radix-ui/react-tabs';
import type { ComponentProps } from 'react';

export const Tabs = TabsPrimitive.Root;

export function TabsList({ className = '', ...props }: ComponentProps<typeof TabsPrimitive.List>) {
  return <TabsPrimitive.List className={`ui-tabs-list ${className}`.trim()} {...props} />;
}

export function TabsTrigger({ className = '', ...props }: ComponentProps<typeof TabsPrimitive.Trigger>) {
  return <TabsPrimitive.Trigger className={`ui-tabs-trigger ${className}`.trim()} {...props} />;
}
