import { forwardRef, type ComponentPropsWithoutRef, type ElementRef } from 'react';
import * as Primitive from '@radix-ui/react-dialog';

export const Root = Primitive.Root;
export const Trigger = Primitive.Trigger;
export const Portal = Primitive.Portal;
export const Close = Primitive.Close;

export const Overlay = forwardRef<ElementRef<typeof Primitive.Overlay>, ComponentPropsWithoutRef<typeof Primitive.Overlay>>(
  ({ className = '', ...props }, ref) => <Primitive.Overlay ref={ref} className={`second-app-dialog-overlay ${className}`} {...props} />,
);
Overlay.displayName = 'DialogOverlay';

export const Content = forwardRef<ElementRef<typeof Primitive.Content>, ComponentPropsWithoutRef<typeof Primitive.Content>>(
  ({ className = '', ...props }, ref) => <Primitive.Content ref={ref} className={`second-app-dialog ${className}`} {...props} />,
);
Content.displayName = 'DialogContent';

export const Title = forwardRef<ElementRef<typeof Primitive.Title>, ComponentPropsWithoutRef<typeof Primitive.Title>>(
  ({ className = '', ...props }, ref) => <Primitive.Title ref={ref} className={`second-app-dialog-title ${className}`} {...props} />,
);
Title.displayName = 'DialogTitle';

export const Description = forwardRef<ElementRef<typeof Primitive.Description>, ComponentPropsWithoutRef<typeof Primitive.Description>>(
  ({ className = '', ...props }, ref) => <Primitive.Description ref={ref} className={`second-app-dialog-description ${className}`} {...props} />,
);
Description.displayName = 'DialogDescription';
