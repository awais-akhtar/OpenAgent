import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { MotionConfig } from 'motion/react';
import { App } from './App';
import './tailwind.css';
import './reference.css';
import './execution-trace.css';
import './app.css';
import { useWorkspaceState } from './state';

class StartupErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    console.error('OpenAgent interface failed to render:', error);
  }

  render() {
    if (this.state.failed) {
      return (
        <main role="alert" style={{ padding: 32, maxWidth: 760, margin: '64px auto', fontFamily: 'system-ui, sans-serif' }}>
          <h1>OpenAgent could not load its interface</h1>
          <p>The local application started, but a frontend error prevented the workspace from rendering.</p>
          <p>Try reloading. If this continues, check the browser developer console and the desktop startup logs.</p>
        </main>
      );
    }
    return this.props.children;
  }
}

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

function Root() {
  const reduceMotion = useWorkspaceState((state) => state.reduceMotion);
  return <QueryClientProvider client={queryClient}>
    <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
      <BrowserRouter><App /></BrowserRouter>
    </MotionConfig>
  </QueryClientProvider>;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StartupErrorBoundary><Root /></StartupErrorBoundary>
  </React.StrictMode>,
);
