import React from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider } from '@arco-design/web-react';
import '@arco-design/web-react/dist/css/arco.css';
import 'virtual:uno.css';
import App from './App';
import './app.css';
import './react-shell.css';

const target = document.getElementById('app');
if (!target) throw new Error('OpenAgent app mount point is missing.');
createRoot(target).render(<React.StrictMode><ConfigProvider size="small" theme={{ primaryColor: '#202020' }}><App /></ConfigProvider></React.StrictMode>);
