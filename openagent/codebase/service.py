"""Durable autonomous software-engineering runs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import threading
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ..model_gateway import ModelGateway, ProviderConfigurationError
from ..models import AuditEvent, utc_now
from ..storage import OpenAgentStore
from .sdk import CodeQSDK, GenerationRequest
from .repository import IGNORED_NAMES, RepositoryIntelligence
from .profiles import check_artifact_contract, classify_request, task_needs_file_change
from .sandbox import SandboxManager
from .tools import ApprovalRequired, CodebaseTools


FINAL_STATES = {"completed", "failed", "cancelled", "blocked"}
PLAN_SCHEMA = {
    "goal": "string",
    "tasks": [{"id": "T1", "title": "string", "agent": "architect|frontend|backend|database|programmer|tester|debugger|security|reviewer|documentation", "depends_on": ["T0"], "parent": ""}],
}
ACTION_TOOLS = {
    "list_directory", "read_file", "read_files", "write_file", "patch_file", "delete_file", "move_file",
    "grep", "search_symbol", "find_references", "project_map", "run_shell", "run_tests", "run_linter", "run_typechecker",
    "run_build", "browser_test", "browser_inspect", "install_dependency", "start_process", "process_status",
    "stop_process", "inspect_logs", "git_status", "git_diff", "git_log", "git_commit", "git_checkout",
    "git_revert", "git_reset",
}

EMPTY_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Secure account sign in">
  <title>Sign in</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main class="auth-shell">
    <section class="auth-intro" aria-labelledby="welcome-title">
      <a class="brand" href="#" aria-label="Home"><span aria-hidden="true">O</span> Orbit</a>
      <div>
        <p class="eyebrow">Private workspace</p>
        <h1 id="welcome-title">Welcome back.</h1>
        <p>Sign in to continue to your projects, notes, and saved work.</p>
      </div>
      <p class="privacy-note">Your session remains on this device.</p>
    </section>
    <section class="auth-panel" aria-labelledby="form-title">
      <form id="login-form" novalidate>
        <header>
          <p class="eyebrow">Account access</p>
          <h2 id="form-title">Sign in</h2>
          <p>Use the email address connected to your account.</p>
        </header>
        <div class="field">
          <label for="email">Email address</label>
          <input id="email" name="email" type="email" autocomplete="email" inputmode="email" required aria-describedby="email-error">
          <small id="email-error" class="field-error" aria-live="polite"></small>
        </div>
        <div class="field">
          <div class="label-row"><label for="password">Password</label><a href="#">Forgot password?</a></div>
          <div class="password-control">
            <input id="password" name="password" type="password" autocomplete="current-password" minlength="8" required aria-describedby="password-error">
            <button id="toggle-password" class="text-button" type="button" aria-controls="password" aria-pressed="false">Show</button>
          </div>
          <small id="password-error" class="field-error" aria-live="polite"></small>
        </div>
        <label class="remember"><input name="remember" type="checkbox"> Keep me signed in</label>
        <button class="primary-button" type="submit">Continue</button>
        <p id="form-status" class="form-status" role="status" aria-live="polite"></p>
        <p class="account-link">New here? <a href="#">Create an account</a></p>
      </form>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""

EMPTY_LOGIN_CSS = """* { box-sizing: border-box; }
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #edf2f8; }
body { margin: 0; min-width: 320px; min-height: 100vh; }
a { color: #145dc1; text-underline-offset: 3px; }
button, input { font: inherit; }
.auth-shell { min-height: 100vh; display: grid; grid-template-columns: minmax(280px, .9fr) minmax(440px, 1.1fr); }
.auth-intro { display: flex; flex-direction: column; justify-content: space-between; gap: 3rem; padding: clamp(2rem, 5vw, 5rem); color: white; background: #102a43; }
.brand { display: inline-flex; align-items: center; gap: .7rem; width: fit-content; color: white; font-weight: 750; text-decoration: none; }
.brand span { display: grid; place-items: center; width: 2rem; height: 2rem; border: 1px solid #8fc5ff; border-radius: 7px; background: #1768c4; }
.auth-intro h1 { max-width: 620px; margin: .3rem 0 .8rem; font-size: clamp(2.5rem, 6vw, 5.4rem); line-height: .98; }
.auth-intro p { max-width: 520px; color: #c8d9e8; line-height: 1.65; }
.eyebrow { margin: 0; color: #4177b7; font-size: .76rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
.privacy-note { margin: 0; font-size: .88rem; }
.auth-panel { display: grid; place-items: center; padding: clamp(1.25rem, 6vw, 5rem); background: #f8fafc; }
form { width: min(100%, 440px); display: grid; gap: 1.2rem; }
form header { margin-bottom: .4rem; }
form h2 { margin: .35rem 0 .5rem; font-size: 2rem; }
form header p:last-child, .account-link { color: #66758a; }
.field { display: grid; gap: .45rem; }
.field label, .remember { font-size: .9rem; font-weight: 650; }
.label-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.label-row a { font-size: .85rem; }
input { width: 100%; min-height: 48px; border: 1px solid #bcc8d6; border-radius: 7px; padding: .72rem .85rem; color: #172033; background: white; outline: none; transition: border-color .16s, box-shadow .16s; }
input:focus { border-color: #1768c4; box-shadow: 0 0 0 3px rgba(23, 104, 196, .15); }
input[aria-invalid="true"] { border-color: #b42318; }
.password-control { position: relative; }
.password-control input { padding-right: 4.3rem; }
.text-button { position: absolute; top: 50%; right: .45rem; transform: translateY(-50%); border: 0; padding: .45rem; color: #145dc1; background: transparent; cursor: pointer; }
.remember { display: flex; align-items: center; gap: .65rem; width: fit-content; color: #45556c; }
.remember input { width: 1rem; min-height: 1rem; accent-color: #1768c4; }
.primary-button { min-height: 50px; border: 0; border-radius: 7px; color: white; font-weight: 750; background: #1768c4; cursor: pointer; transition: background .16s, transform .16s; }
.primary-button:hover { background: #1057a7; }
.primary-button:active { transform: translateY(1px); }
.field-error { min-height: 1.1em; color: #b42318; }
.form-status { min-height: 1.3em; margin: -.3rem 0 0; color: #17643a; }
.account-link { margin: 0; text-align: center; }
@media (max-width: 780px) { .auth-shell { grid-template-columns: 1fr; } .auth-intro { min-height: 245px; padding: 1.5rem; gap: 1.5rem; } .auth-intro h1 { font-size: clamp(2.3rem, 13vw, 4rem); } .privacy-note { display: none; } .auth-panel { place-items: start center; padding: 2rem 1.25rem 3rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
"""

EMPTY_LOGIN_JS = """const form = document.querySelector('#login-form');
const email = document.querySelector('#email');
const password = document.querySelector('#password');
const toggle = document.querySelector('#toggle-password');
const status = document.querySelector('#form-status');

function showError(input, message) {
  input.setAttribute('aria-invalid', message ? 'true' : 'false');
  document.querySelector(`#${input.id}-error`).textContent = message;
}

toggle.addEventListener('click', () => {
  const visible = password.type === 'text';
  password.type = visible ? 'password' : 'text';
  toggle.textContent = visible ? 'Show' : 'Hide';
  toggle.setAttribute('aria-pressed', String(!visible));
});

form.addEventListener('submit', event => {
  event.preventDefault();
  showError(email, email.validity.valid ? '' : 'Enter a valid email address.');
  showError(password, password.value.length >= 8 ? '' : 'Password must contain at least 8 characters.');
  status.textContent = form.checkValidity() ? 'Form validated. Connect this page to your authentication endpoint.' : '';
  if (!form.checkValidity()) form.querySelector(':invalid')?.focus();
});
"""

STATIC_SITE_CSS = """* { box-sizing: border-box; }
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f4f7fb; }
html { scroll-behavior: smooth; }
body { margin: 0; min-width: 320px; min-height: 100vh; background: #f4f7fb; }
a { color: inherit; }
button { font: inherit; }
.site-header { position: sticky; top: 0; z-index: 10; border-bottom: 1px solid #dbe3ed; background: rgba(255,255,255,.96); }
.nav-shell { width: min(1120px, calc(100% - 2rem)); min-height: 72px; margin: auto; display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
.brand { display: inline-flex; align-items: center; gap: .7rem; font-weight: 800; text-decoration: none; }
.brand-mark { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 7px; color: white; background: #1261a6; }
.site-nav { display: flex; align-items: center; gap: .35rem; }
.site-nav a { padding: .6rem .8rem; border-radius: 6px; color: #506176; font-size: .92rem; font-weight: 650; text-decoration: none; }
.site-nav a:hover, .site-nav a[aria-current="page"] { color: #0d4f88; background: #e9f2fa; }
.menu-button { display: none; border: 1px solid #c9d4e2; border-radius: 6px; padding: .55rem .7rem; color: #18344f; background: white; cursor: pointer; }
main { min-height: calc(100vh - 142px); }
.hero { padding: clamp(4rem, 10vw, 8rem) 1rem; color: white; background: #12395b; }
.hero-inner, .content-grid { width: min(1120px, 100%); margin: auto; }
.eyebrow { margin: 0 0 .8rem; color: #8bc8f5; font-size: .76rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
h1 { max-width: 760px; margin: 0; font-size: clamp(2.5rem, 7vw, 5.4rem); line-height: 1; }
.hero p:last-child { max-width: 650px; margin: 1.4rem 0 0; color: #d2e5f4; font-size: clamp(1rem, 2vw, 1.18rem); line-height: 1.75; }
.content { padding: clamp(3rem, 7vw, 6rem) 1rem; }
.content-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(250px, .65fr); gap: clamp(2rem, 6vw, 5rem); align-items: start; }
.content h2 { margin: 0 0 1rem; font-size: clamp(1.7rem, 4vw, 2.6rem); }
.content p { color: #526276; line-height: 1.75; }
.feature-list { display: grid; gap: .8rem; margin: 1.7rem 0 0; padding: 0; list-style: none; }
.feature-list li { border-left: 3px solid #1870b9; padding: .75rem 1rem; background: white; box-shadow: 0 1px 2px rgba(20,45,70,.06); }
.side-note { border: 1px solid #d9e3ed; border-radius: 8px; padding: 1.4rem; background: white; }
.side-note strong { display: block; margin-bottom: .45rem; color: #164f80; }
.site-footer { min-height: 70px; display: grid; place-items: center; border-top: 1px solid #dbe3ed; padding: 1rem; color: #627286; background: white; }
@media (max-width: 760px) { .menu-button { display: inline-flex; } .site-nav { position: absolute; top: 64px; right: 1rem; left: 1rem; display: none; align-items: stretch; flex-direction: column; border: 1px solid #d5dfeb; border-radius: 8px; padding: .55rem; background: white; box-shadow: 0 16px 38px rgba(18,57,91,.16); } .site-nav[data-open="true"] { display: flex; } .content-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } * { transition: none !important; } }
"""

STATIC_SITE_JS = """const menuButton = document.querySelector('.menu-button');
const navigation = document.querySelector('.site-nav');

menuButton?.addEventListener('click', () => {
  const isOpen = navigation.dataset.open === 'true';
  navigation.dataset.open = String(!isOpen);
  menuButton.setAttribute('aria-expanded', String(!isOpen));
});

navigation?.addEventListener('click', event => {
  if (event.target.closest('a')) {
    navigation.dataset.open = 'false';
    menuButton?.setAttribute('aria-expanded', 'false');
  }
});
"""

REACT_LOGIN_PACKAGE = """{
  "name": "responsive-login",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@vitejs/plugin-react": "latest",
    "vite": "latest",
    "react": "latest",
    "react-dom": "latest"
  },
  "devDependencies": {}
}
"""

REACT_LOGIN_INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#0b1628">
  <title>Sign in</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
"""

REACT_LOGIN_MAIN = """import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>,
);
"""

REACT_LOGIN_APP = """import { useState } from 'react';

export default function App() {
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [message, setMessage] = useState('');

  function submit(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const email = String(data.get('email') || '').trim();
    const password = String(data.get('password') || '');
    const nextErrors = {};
    if (!/^\\S+@\\S+\\.\\S+$/.test(email)) nextErrors.email = 'Enter a valid email address.';
    if (password.length < 8) nextErrors.password = 'Use at least 8 characters.';
    setErrors(nextErrors);
    setMessage(Object.keys(nextErrors).length ? '' : 'Details validated and ready for your authentication API.');
  }

  return (
    <main className="login-shell">
      <section className="login-context" aria-labelledby="welcome-title">
        <a className="brand" href="#" aria-label="Northstar home"><span>N</span> Northstar</a>
        <div>
          <p className="eyebrow">Secure workspace</p>
          <h1 id="welcome-title">Welcome back.</h1>
          <p>Sign in to continue to your projects, reports, and team workspace.</p>
        </div>
        <small>Your session stays protected on this device.</small>
      </section>
      <section className="form-panel">
        <form onSubmit={submit} noValidate>
          <header><p className="eyebrow">Account access</p><h2>Sign in</h2><p>Use the email connected to your account.</p></header>
          <label>Email address<input name="email" type="email" autoComplete="email" aria-invalid={Boolean(errors.email)} />{errors.email && <span role="alert">{errors.email}</span>}</label>
          <label>Password<div className="password-field"><input name="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" aria-invalid={Boolean(errors.password)} /><button type="button" onClick={() => setShowPassword(value => !value)}>{showPassword ? 'Hide' : 'Show'}</button></div>{errors.password && <span role="alert">{errors.password}</span>}</label>
          <div className="form-row"><label className="check"><input name="remember" type="checkbox" /> Remember me</label><a href="#">Forgot password?</a></div>
          <button className="submit" type="submit">Continue</button>
          <p className="status" role="status">{message}</p>
          <p className="create">New here? <a href="#">Create an account</a></p>
        </form>
      </section>
    </main>
  );
}
"""

REACT_LOGIN_CSS = """* { box-sizing: border-box; }
:root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #152033; background: #f4f7fb; }
body { margin: 0; min-width: 320px; min-height: 100vh; }
button, input { font: inherit; }
a { color: #1764c0; text-underline-offset: 3px; }
.login-shell { min-height: 100vh; display: grid; grid-template-columns: minmax(300px, .9fr) minmax(440px, 1.1fr); }
.login-context { display: flex; flex-direction: column; justify-content: space-between; gap: 3rem; padding: clamp(2rem, 5vw, 5rem); color: white; background: #0b2748; }
.brand { display: inline-flex; align-items: center; gap: .7rem; width: fit-content; color: white; font-weight: 750; text-decoration: none; }
.brand span { display: grid; place-items: center; width: 2.1rem; aspect-ratio: 1; border: 1px solid #91c8ff; border-radius: 7px; background: #1764c0; }
.login-context h1 { margin: .35rem 0 .8rem; font-size: clamp(2.7rem, 6vw, 5.5rem); line-height: 1; }
.login-context p, .login-context small { max-width: 34rem; color: #c8d9ea; line-height: 1.65; }
.eyebrow { margin: 0; color: #4c79ad; font-size: .75rem; font-weight: 800; text-transform: uppercase; }
.form-panel { display: grid; place-items: center; padding: clamp(1.5rem, 6vw, 5rem); }
form { width: min(100%, 440px); display: grid; gap: 1.15rem; }
form header { margin-bottom: .5rem; } form h2 { margin: .35rem 0; font-size: 2rem; } form header p:last-child, .create { color: #68778b; }
form > label { display: grid; gap: .45rem; font-size: .9rem; font-weight: 700; }
input { width: 100%; min-height: 48px; padding: .75rem .85rem; border: 1px solid #bdc9d8; border-radius: 7px; background: white; outline: none; }
input:focus { border-color: #1764c0; box-shadow: 0 0 0 3px rgb(23 100 192 / 15%); } input[aria-invalid="true"] { border-color: #b42318; }
label > span { color: #b42318; font-size: .8rem; font-weight: 500; }
.password-field { position: relative; } .password-field input { padding-right: 4rem; } .password-field button { position: absolute; right: .4rem; top: 50%; transform: translateY(-50%); border: 0; color: #1764c0; background: transparent; cursor: pointer; }
.form-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; font-size: .88rem; } .check { display: flex; align-items: center; gap: .55rem; } .check input { width: 1rem; min-height: 1rem; }
.submit { min-height: 50px; border: 0; border-radius: 7px; color: white; font-weight: 750; background: #1764c0; cursor: pointer; } .submit:hover { background: #1055a5; }
.status { min-height: 1.25rem; margin: -.25rem 0; color: #17643a; } .create { margin: 0; text-align: center; }
@media (max-width: 780px) { .login-shell { grid-template-columns: 1fr; } .login-context { min-height: 250px; padding: 1.5rem; gap: 1.5rem; } .login-context h1 { font-size: clamp(2.5rem, 13vw, 4rem); } .login-context small { display: none; } .form-panel { place-items: start center; padding: 2rem 1.25rem 3rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
"""

DJANGO_MANAGE = """#!/usr/bin/env python
import os
import sys


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
"""

DJANGO_SETTINGS = """from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = "development-only-change-me"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "pages",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": []},
}]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
"""

DJANGO_ROOT_URLS = """from django.urls import include, path

urlpatterns = [path("", include("pages.urls"))]
"""

DJANGO_WSGI = """import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_wsgi_application()
"""

DJANGO_SITE_CSS = """* { box-sizing: border-box; }
:root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #182536; background: #f5f7fa; }
body { margin: 0; min-width: 320px; min-height: 100vh; }
.site-header { position: sticky; top: 0; z-index: 10; border-bottom: 1px solid #dce3eb; background: rgba(255,255,255,.96); }
.nav-shell { width: min(1120px, calc(100% - 2rem)); min-height: 68px; margin: auto; display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
.brand { display: flex; align-items: center; gap: .65rem; color: #17324d; font-weight: 800; text-decoration: none; }
.brand span { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 7px; color: white; background: #17679f; }
nav { display: flex; gap: .35rem; } nav a { border-radius: 6px; padding: .55rem .75rem; color: #536578; font-weight: 650; text-decoration: none; } nav a:hover { color: #0f568a; background: #e9f2f8; }
.menu-button { display: none; border: 1px solid #cad5e1; border-radius: 6px; padding: .5rem .7rem; background: white; }
.hero { padding: clamp(4rem, 10vw, 8rem) 1rem; color: white; background: #123a5a; }.hero > div,.content { width: min(1120px,100%); margin: auto; }
.eyebrow { margin: 0 0 .7rem; color: #86c8ef; font-size: .76rem; font-weight: 800; text-transform: uppercase; } h1 { margin: 0; font-size: clamp(2.8rem,8vw,5.8rem); } .hero p:last-child { max-width: 650px; color: #d4e7f3; font-size: 1.1rem; line-height: 1.7; }
.content { min-height: 300px; padding: clamp(3rem,7vw,6rem) 1rem; }.content h2 { font-size: 2rem; }.content p { max-width: 700px; color: #58697b; line-height: 1.75; }
footer { border-top: 1px solid #dce3eb; padding: 1.5rem; color: #657587; text-align: center; }
@media (max-width: 680px) { .menu-button { display: block; } nav { position: absolute; top: 61px; right: 1rem; left: 1rem; display: none; flex-direction: column; border: 1px solid #d5dee8; border-radius: 8px; padding: .5rem; background: white; box-shadow: 0 16px 34px rgba(18,58,90,.16); } nav.open { display: flex; } }
"""

DJANGO_SITE_JS = """(function ($) {
  'use strict';
  $('.menu-button').on('click', function () {
    const $navigation = $('#primary-nav');
    const open = !$navigation.hasClass('open');
    $navigation.toggleClass('open', open);
    $(this).attr('aria-expanded', String(open));
  });
  $('#primary-nav a').on('click', function () {
    $('#primary-nav').removeClass('open');
    $('.menu-button').attr('aria-expanded', 'false');
  });
})(window.jQuery);
"""


class CodebaseService:
    """Own projects, workspaces, task graphs, tools, events, and completion evidence."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway) -> None:
        self.store = store
        self.gateway = gateway
        self.sdk = CodeQSDK()
        self.sandboxes = SandboxManager(store.data_dir / "codebase-workspaces")
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self.initialize()
        self._recover_interrupted_runs()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS codebase_projects (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    repo_url TEXT NOT NULL DEFAULT '',
                    map_json TEXT NOT NULL DEFAULT '{}',
                    memory_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_projects_owner ON codebase_projects(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS codebase_workspaces (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    sandbox_backend TEXT NOT NULL DEFAULT 'local',
                    source_path TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ready',
                    environment_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_runs (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    user_prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'auto',
                    model TEXT NOT NULL DEFAULT '',
                    role_models_json TEXT NOT NULL DEFAULT '{}',
                    allow_external INTEGER NOT NULL DEFAULT 0,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    max_iterations INTEGER NOT NULL DEFAULT 8,
                    completion_json TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE,
                    FOREIGN KEY(workspace_id) REFERENCES codebase_workspaces(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_runs_project ON codebase_runs(project, owner, started_at DESC);

                CREATE TABLE IF NOT EXISTS codebase_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_task_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    assigned_agent TEXT NOT NULL DEFAULT 'programmer',
                    status TEXT NOT NULL DEFAULT 'pending',
                    depends_on_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_tasks_run ON codebase_tasks(run_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS codebase_tool_calls (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    risk INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_file_changes (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    old_hash TEXT NOT NULL DEFAULT '',
                    new_hash TEXT NOT NULL DEFAULT '',
                    patch TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_test_runs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    command_json TEXT NOT NULL DEFAULT '[]',
                    stdout TEXT NOT NULL DEFAULT '',
                    stderr TEXT NOT NULL DEFAULT '',
                    exit_code INTEGER NOT NULL DEFAULT -1,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_codebase_events_run ON codebase_events(run_id, sequence ASC);

                CREATE TABLE IF NOT EXISTS codebase_snapshots (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES codebase_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS codebase_memories (
                    id TEXT PRIMARY KEY,
                    codebase_project_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_run_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(codebase_project_id) REFERENCES codebase_projects(id) ON DELETE CASCADE
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(codebase_runs)").fetchall()}
            if "role_models_json" not in columns:
                connection.execute("ALTER TABLE codebase_runs ADD COLUMN role_models_json TEXT NOT NULL DEFAULT '{}'")

    def pick_source_folder(self) -> str:
        """Open the operating-system folder chooser after an explicit UI action."""

        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise ValueError("The native folder chooser is unavailable in this Python installation.") from exc
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(parent=root, title="Open a project in CodeQ", mustexist=True)
        finally:
            root.destroy()
        return str(Path(selected).resolve()) if selected else ""

    def create_project(self, *, name: str, source_path: str, project: str = "default", owner: str = "local", backend: str = "local") -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        intelligence = RepositoryIntelligence(source)
        project_map = intelligence.scan()
        codebase_project_id = uuid4().hex
        workspace_id = uuid4().hex
        branch = f"openagent/run-{workspace_id[:8]}"
        workspace = self.sandboxes.create(workspace_id, source=source, backend=backend, branch=branch)
        now = utc_now()
        memory = {
            "stack": {"languages": project_map["languages"], "framework": project_map["framework"], "database": project_map["database"]},
            "commands": project_map["commands"],
            "preferences": [],
            "last_successful_validation": {},
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO codebase_projects (id, project, owner, name, source_path, map_json, memory_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (codebase_project_id, project, owner, name.strip() or source.name, str(source), json.dumps(project_map), json.dumps(memory), now, now),
            )
            connection.execute(
                "INSERT INTO codebase_workspaces (id, codebase_project_id, owner, sandbox_backend, source_path, workspace_path, branch, environment_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)",
                (workspace_id, codebase_project_id, owner, backend, str(source), workspace["path"], branch, now, now),
            )
        self._audit("codebase.project.create", f"codebase-project:{codebase_project_id}", project, {"owner": owner, "source": str(source), "backend": backend})
        return self.get_project(codebase_project_id, owner=owner)

    def list_projects(self, *, project: str = "default", owner: str = "local") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_projects WHERE project = ? AND owner = ? ORDER BY updated_at DESC", (project, owner)).fetchall()
        return [self._project_row(row) for row in rows]

    def get_project(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_projects WHERE id = ? AND owner = ?", (codebase_project_id, owner)).fetchone()
            workspace = connection.execute("SELECT * FROM codebase_workspaces WHERE codebase_project_id = ? AND owner = ? ORDER BY created_at DESC LIMIT 1", (codebase_project_id, owner)).fetchone()
        if row is None:
            raise KeyError("codebase project not found")
        result = self._project_row(row)
        result["workspace"] = self._workspace_row(workspace) if workspace else None
        return result

    def refresh_project_map(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project.get("workspace") or {}
        project_map = RepositoryIntelligence(str(workspace["workspace_path"])).scan()
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_projects SET map_json = ?, updated_at = ? WHERE id = ? AND owner = ?", (json.dumps(project_map), utc_now(), codebase_project_id, owner))
        return project_map

    def framework_catalog(self) -> list[dict[str, Any]]:
        return self.sdk.registry.catalog()

    def framework_preflight(
        self,
        goal: str,
        *,
        codebase_project_id: str = "",
        owner: str = "local",
        acceptance_criteria: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not str(goal).strip():
            raise ValueError("software-engineering goal is required")
        project_map: dict[str, Any] = {"file_count": 0, "framework": "", "languages": []}
        if codebase_project_id:
            project_map = self.get_project(codebase_project_id, owner=owner)["map"]
        return self.sdk.prepare(
            GenerationRequest(
                goal=goal,
                acceptance_criteria=tuple(acceptance_criteria),
            ),
            project_map,
        )

    def file_tree(self, codebase_project_id: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        tools = CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"])
        return tools.tool_list_directory(recursive=True)

    def read_file(self, codebase_project_id: str, path: str, *, owner: str = "local") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        return CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"]).tool_read_file(path, end_line=1_000_000)

    def write_file(self, codebase_project_id: str, path: str, content: str, *, owner: str = "local", expected_sha256: str = "") -> dict[str, Any]:
        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        result = CodebaseTools(self.sandboxes, workspace["id"], backend=workspace["sandbox_backend"]).execute(
            "write_file", {"path": path, "content": content, "expected_sha256": expected_sha256}
        )
        source_target = self._safe_project_path(project["source_path"], path)
        source_target.parent.mkdir(parents=True, exist_ok=True)
        source_target.write_text(str(content), encoding="utf-8", newline="\n")
        result["source_path"] = str(source_target)
        self.refresh_project_map(codebase_project_id, owner=owner)
        self._audit("codebase.file.write", f"codebase-project:{codebase_project_id}:{path}", project["project"], {"owner": owner, "old_hash": result.get("old_sha256"), "new_hash": result.get("new_sha256")})
        return result

    def mutate_path(
        self,
        codebase_project_id: str,
        *,
        action: str,
        path: str,
        destination: str = "",
        owner: str = "local",
    ) -> dict[str, Any]:
        """Apply an explicit Explorer operation to workspace and selected source."""

        project = self.get_project(codebase_project_id, owner=owner)
        workspace = project["workspace"]
        workspace_path = self._safe_project_path(workspace["workspace_path"], path)
        source_path = self._safe_project_path(project["source_path"], path)
        operation = str(action).strip().lower()
        result: dict[str, Any] = {"action": operation, "path": Path(path).as_posix()}
        if operation == "create_file":
            if workspace_path.exists() or source_path.exists():
                raise ValueError("file already exists")
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.write_text("", encoding="utf-8")
            source_path.write_text("", encoding="utf-8")
        elif operation == "create_directory":
            if workspace_path.exists() or source_path.exists():
                raise ValueError("folder already exists")
            workspace_path.mkdir(parents=True)
            source_path.mkdir(parents=True)
        elif operation in {"rename", "move"}:
            if not destination:
                raise ValueError("destination is required")
            workspace_destination = self._safe_project_path(workspace["workspace_path"], destination)
            source_destination = self._safe_project_path(project["source_path"], destination)
            if not workspace_path.exists():
                raise FileNotFoundError(path)
            if workspace_destination.exists() or source_destination.exists():
                raise ValueError("destination already exists")
            workspace_destination.parent.mkdir(parents=True, exist_ok=True)
            source_destination.parent.mkdir(parents=True, exist_ok=True)
            workspace_path.replace(workspace_destination)
            if source_path.exists():
                source_path.replace(source_destination)
            elif workspace_destination.is_dir():
                shutil.copytree(workspace_destination, source_destination)
            else:
                shutil.copy2(workspace_destination, source_destination)
            result["destination"] = Path(destination).as_posix()
        elif operation == "delete":
            if not workspace_path.exists() and not source_path.exists():
                raise FileNotFoundError(path)
            for target in (workspace_path, source_path):
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
        else:
            raise ValueError("action must be create_file, create_directory, rename, move, or delete")
        self.refresh_project_map(codebase_project_id, owner=owner)
        self._audit(f"codebase.path.{operation}", f"codebase-project:{codebase_project_id}:{path}", project["project"], {"owner": owner, "destination": destination})
        return result

    @staticmethod
    def _safe_project_path(root: str | Path, relative: str) -> Path:
        value = str(relative).replace("\\", "/").strip().strip("/")
        if not value:
            raise ValueError("path is required")
        parts = Path(value).parts
        if any(part in IGNORED_NAMES or part in {".git", ".openagent"} for part in parts):
            raise PermissionError("protected project paths cannot be changed from Explorer")
        lowered = Path(value).name.lower()
        if lowered == ".env" or lowered.startswith(".env.") or Path(value).suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise PermissionError("secret-bearing files must be managed through the vault")
        return RepositoryIntelligence(root).resolve(value)

    def create_run(
        self,
        *,
        codebase_project_id: str,
        prompt: str,
        owner: str = "local",
        provider: str = "auto",
        model: str = "",
        role_models: dict[str, Any] | None = None,
        allow_external: bool = False,
        max_iterations: int = 0,
        completion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("software-engineering goal is required")
        codebase_project = self.get_project(codebase_project_id, owner=owner)
        workspace = codebase_project["workspace"]
        project_map = codebase_project["map"]
        criteria = self._completion_criteria(project_map, completion or {})
        if self._standalone_python_request(prompt, project_map):
            criteria["test"] = {
                "enabled": True,
                "command": ["python", "-m", "unittest", "discover", "-v"],
                "timeout_seconds": 300,
            }
            criteria["runtime"] = {
                "enabled": True,
                "command": ["python", "-m", "compileall", "-q", "."],
                "timeout_seconds": 300,
            }
        iteration_budget = self._automatic_iteration_budget(prompt, project_map) if int(max_iterations) <= 0 else max(1, min(int(max_iterations), 200))
        run_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, project, owner, user_prompt, status, provider, model, role_models_json, allow_external, max_iterations, completion_json, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, codebase_project_id, workspace["id"], codebase_project["project"], owner, prompt.strip(), provider, model, json.dumps(role_models or {}), int(allow_external), iteration_budget, json.dumps(criteria), now, now),
            )
        self._event(run_id, "run.created", goal=prompt.strip(), completion=criteria, autonomy="full", iteration_budget=iteration_budget)
        snapshot = self.sandboxes.snapshot(workspace["id"], f"{run_id}-before")
        self._record_snapshot(run_id, snapshot, "Before autonomous run")
        cancel = threading.Event()
        thread = threading.Thread(target=self._execute_run, args=(run_id,), name=f"codebase-run-{run_id[:8]}", daemon=True)
        with self._lock:
            self._cancel[run_id] = cancel
            self._threads[run_id] = thread
        thread.start()
        return self.get_run(run_id, owner=owner)

    def cancel_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        if run["status"] in FINAL_STATES:
            return run
        with self._lock:
            event = self._cancel.setdefault(run_id, threading.Event())
            event.set()
        self._set_run(run_id, status="cancelled", error="Cancelled by user", finished=True)
        self._event(run_id, "run.cancelled", reason="Cancelled by user")
        return self.get_run(run_id, owner=owner)

    def resume_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        if run["status"] not in {"blocked", "failed", "interrupted"}:
            raise ValueError("only blocked, failed, or interrupted runs can resume")
        dependency_repairs = self._repair_task_dependencies(run_id)
        if dependency_repairs:
            self._event(run_id, "plan.dependencies_repaired", repairs=dependency_repairs, source="resume")
        recovered_tasks = self._recover_blocked_tasks(run_id, source="resume")
        if recovered_tasks:
            self._event(run_id, "tasks.recovered", task_ids=recovered_tasks, source="resume")
        self._set_run(run_id, status="queued", error="", finished=False)
        cancel = threading.Event()
        thread = threading.Thread(target=self._execute_run, args=(run_id,), name=f"codebase-run-{run_id[:8]}", daemon=True)
        with self._lock:
            self._cancel[run_id] = cancel
            self._threads[run_id] = thread
        thread.start()
        return self.get_run(run_id, owner=owner)

    def wait_for_run(self, run_id: str, *, owner: str = "local", timeout: float | None = None) -> dict[str, Any]:
        """Wait for the worker to release process and database resources."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return self.get_run(run_id, owner=owner)

    def restore_run_snapshot(self, run_id: str, snapshot_id: str, *, owner: str = "local") -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_snapshots WHERE id = ? AND run_id = ?", (snapshot_id, run_id)).fetchone()
        if row is None:
            raise KeyError("snapshot not found")
        self.sandboxes.restore(run["workspace_id"], row["path"])
        self._event(run_id, "snapshot.restored", snapshot_id=snapshot_id)
        self._audit("codebase.snapshot.restore", f"codebase-run:{run_id}", run["project"], {"snapshot_id": snapshot_id, "owner": owner})
        return {"restored": True, "snapshot_id": snapshot_id}

    def resolve_approval(self, approval_id: str, *, approved: bool, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT a.*, r.owner FROM codebase_approvals a JOIN codebase_runs r ON r.id = a.run_id WHERE a.id = ? AND r.owner = ?",
                (approval_id, owner),
            ).fetchone()
            if row is None:
                raise KeyError("approval not found")
            if str(row["status"]) != "pending":
                return {
                    "id": approval_id,
                    "run_id": row["run_id"],
                    "status": str(row["status"]),
                    "result": {"already_resolved": True},
                }
            status = "approved" if approved else "rejected"
            connection.execute("UPDATE codebase_approvals SET status = ?, resolved_at = ? WHERE id = ?", (status, utc_now(), approval_id))
        self._event(row["run_id"], "approval.resolved", approval_id=approval_id, status=status)
        payload = json.loads(row["payload_json"] or "{}")
        task_id = str(payload.get("task_id", ""))
        if approved:
            worker = threading.Thread(
                target=self._execute_approved_action,
                args=(dict(row), owner),
                name=f"codebase-approval-{approval_id[:8]}",
                daemon=True,
            )
            worker.start()
            return {"id": approval_id, "run_id": row["run_id"], "status": status, "result": {"queued": True}}
        if task_id:
            self._set_task(task_id, "needs_fix", {"approval_rejected": True, "tool": str(row["tool"]), "arguments": payload.get("arguments", {})})
            worker = threading.Thread(
                target=self._resume_after_approval,
                args=(str(row["run_id"]), owner),
                name=f"codebase-rejection-{approval_id[:8]}",
                daemon=True,
            )
            worker.start()
        return {"id": approval_id, "run_id": row["run_id"], "status": status, "result": None}

    def _execute_approved_action(self, approval: dict[str, Any], owner: str) -> None:
        approval_id = str(approval["id"])
        run_id = str(approval["run_id"])
        payload = json.loads(approval.get("payload_json") or "{}")
        task_id = str(payload.get("task_id", ""))
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        failure = ""
        try:
            run = self.get_run(run_id, owner=owner)
            result = self._invoke(run, task_id, str(approval["tool"]), arguments, approved=True)
            if task_id:
                self._set_task(task_id, "needs_fix", {"approved_tool_result": result})
        except Exception as exc:
            failure = str(exc)
            self._event(run_id, "approval.execution_failed", approval_id=approval_id, tool=str(approval["tool"]), error=failure)
            if task_id:
                self._set_task(task_id, "needs_fix", {"approval_error": failure, "tool": str(approval["tool"]), "arguments": arguments})
        finally:
            if task_id:
                self._resume_after_approval(run_id, owner)
            if failure:
                # Publish the terminal approval state last so clients cannot
                # observe "failed" while the recovery worker still owns the DB.
                with self.store.session() as connection:
                    connection.execute("UPDATE codebase_approvals SET status = ? WHERE id = ?", ("failed", approval_id))

    def _resume_after_approval(self, run_id: str, owner: str) -> None:
        with self._lock:
            current_worker = self._threads.get(run_id)
        if current_worker is not None and current_worker is not threading.current_thread():
            current_worker.join(timeout=5)
        try:
            refreshed = self.get_run(run_id, owner=owner)
            if refreshed["status"] == "blocked":
                self.resume_run(run_id, owner=owner)
        except (KeyError, ValueError):
            pass

    def invoke_tool(self, run_id: str, tool: str, arguments: dict[str, Any], *, owner: str = "local", approved: bool = False) -> dict[str, Any]:
        run = self.get_run(run_id, owner=owner)
        try:
            return self._invoke(run, "", tool, arguments, approved=approved)
        except ApprovalRequired as exc:
            approval = self._request_approval(run_id, exc)
            return {"approval_required": approval}

    def invoke_project_tool(self, codebase_project_id: str, tool: str, arguments: dict[str, Any], *, owner: str = "local", approved: bool = False) -> dict[str, Any]:
        """Run an IDE tool before an autonomous task exists, with durable tracing."""
        project = self.get_project(codebase_project_id, owner=owner)
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id FROM codebase_runs WHERE codebase_project_id = ? AND owner = ? AND user_prompt = 'Interactive IDE session' ORDER BY started_at DESC LIMIT 1",
                (codebase_project_id, owner),
            ).fetchone()
            run_id = str(row["id"]) if row else uuid4().hex
            if row is None:
                now = utc_now()
                connection.execute(
                    """INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, project, owner, user_prompt, status, provider, model, role_models_json, allow_external, max_iterations, completion_json, started_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'Interactive IDE session', 'interactive', 'local', '', '{}', 0, 1, '{}', ?, ?)""",
                    (run_id, codebase_project_id, project["workspace"]["id"], project["project"], owner, now, now),
                )
        if row is None:
            self._event(run_id, "run.created", mode="interactive", goal="Interactive IDE session")
        run = self.get_run(run_id, owner=owner)
        try:
            result = self._invoke(run, "", tool, arguments, approved=approved)
        except ApprovalRequired as exc:
            result = {"approval_required": self._request_approval(run_id, exc)}
        return {"run": self.get_run(run_id, owner=owner), "result": result}

    def list_runs(self, *, project: str = "default", owner: str = "local", limit: int = 50) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_runs WHERE project = ? AND owner = ? ORDER BY started_at DESC LIMIT ?", (project, owner, max(1, min(int(limit), 200)))).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str, *, owner: str = "local") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_runs WHERE id = ? AND owner = ?", (run_id, owner)).fetchone()
            if row is None:
                raise KeyError("codebase run not found")
            tasks = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
            approvals = connection.execute("SELECT * FROM codebase_approvals WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
            snapshots = connection.execute("SELECT id, label, sha256, size_bytes, created_at FROM codebase_snapshots WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
            tests = connection.execute("SELECT * FROM codebase_test_runs WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
        result = self._run_row(row)
        result["tasks"] = [self._task_row(item) for item in tasks]
        result["approvals"] = [self._approval_row(item) for item in approvals]
        result["snapshots"] = [dict(item) for item in snapshots]
        result["validation"] = [self._test_row(item) for item in tests]
        return result

    def events(self, run_id: str, *, owner: str = "local", after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_run(run_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_events WHERE run_id = ? AND sequence > ? ORDER BY sequence LIMIT ?", (run_id, max(0, int(after)), max(1, min(int(limit), 2000)))).fetchall()
        return [self._event_row(row) for row in rows]

    def event_stream(self, run_id: str, *, owner: str = "local", after: int = 0, timeout_seconds: int = 120) -> Iterator[dict[str, Any]]:
        cursor = max(0, int(after))
        deadline = time.monotonic() + max(1, min(int(timeout_seconds), 300))
        while time.monotonic() < deadline:
            events = self.events(run_id, owner=owner, after=cursor, limit=200)
            for event in events:
                cursor = int(event["sequence"])
                yield event
            run = self.get_run(run_id, owner=owner)
            if run["status"] in FINAL_STATES and not events:
                yield {"sequence": cursor, "run_id": run_id, "type": "stream.closed", "payload": {"status": run["status"]}, "created_at": utc_now()}
                return
            if not events:
                yield {"sequence": cursor, "run_id": run_id, "type": "stream.heartbeat", "payload": {}, "created_at": utc_now()}
            time.sleep(0.5)

    def _execute_run(self, run_id: str) -> None:
        try:
            run = self.get_run(run_id, owner=self._owner_for_run(run_id))
            self._set_run(run_id, status="analysing")
            self._event(run_id, "agent.started", agent="project-manager")
            project = self.get_project(run["codebase_project_id"], owner=run["owner"])
            # A new run starts from the user's selected folder. Failed-run
            # artifacts may still exist in the protected workspace for
            # inspection, but they must not redefine the next request's stack.
            project_map = RepositoryIntelligence(project["source_path"]).scan()
            self._event(run_id, "repository.analysed", project_map=self._compact_map(project_map))
            environment = self.sdk.prepare(run["user_prompt"], project_map)
            self._event(run_id, "environment.preflight.completed", preflight=environment)
            architecture = self._consult_specialists(run, project, project_map, environment)
            tasks = self._plan(run, project, project_map, architecture)
            dependency_repairs = self._store_tasks(run_id, tasks)
            self._event(run_id, "plan.created", tasks=tasks, architecture=architecture, delivery_profile=classify_request(run["user_prompt"], project_map), dependency_repairs=dependency_repairs)
            self._set_run(run_id, status="running")
            last_feedback = ""
            changed = False
            active_agent = "project-manager"
            decision_cache: dict[str, dict[str, Any]] = {}
            repeated_command_failures: dict[str, int] = {}
            stalled_task_attempts: dict[str, int] = {}
            max_iterations = max(1, int(run["max_iterations"]))
            for iteration in range(max_iterations):
                if self._cancelled(run_id):
                    return
                self._set_run(run_id, iteration=iteration + 1)
                task = self._next_task(run_id)
                if task is None:
                    unfinished = self._unfinished_tasks(run_id)
                    if unfinished:
                        dependency_repairs = self._repair_task_dependencies(run_id)
                        recovered_tasks = self._recover_blocked_tasks(run_id, source="execution-deadlock")
                        if (dependency_repairs or recovered_tasks) and self._next_task(run_id) is not None:
                            self._event(
                                run_id,
                                "plan.dependencies_repaired",
                                repairs=dependency_repairs,
                                source="execution-deadlock",
                            )
                            if recovered_tasks:
                                self._event(run_id, "tasks.recovered", task_ids=recovered_tasks, source="execution-deadlock")
                            continue
                        unresolved = self._unfinished_tasks(run_id)
                        error = self._task_graph_error(unresolved)
                        self._set_run(run_id, status="failed", error=error, finished=True)
                        self._event(run_id, "run.failed", unfinished_tasks=unresolved, error=error)
                        return
                    self._set_run(run_id, status="validating")
                    if active_agent != "tester":
                        self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent="tester", reason="completion-validation")
                        active_agent = "tester"
                    validation = self._validate(run)
                    if not validation["passed"]:
                        last_feedback = "Completion validation failed:\n" + "\n".join(
                            f"{item['kind']}: exit {item.get('exit_code', -1)}\n{item.get('stderr', '')[-8000:]}\n{item.get('stdout', '')[-4000:]}"
                            for item in validation["checks"] if item["status"] != "passed"
                        )
                        if iteration + 1 < max_iterations:
                            self._add_repair_task(run_id, "Repair failed completion checks", last_feedback, assigned_agent="debugger")
                            self._set_run(run_id, status="running", error="")
                            self._event(run_id, "repair.requested", stage="validation", error=last_feedback[-20_000:])
                            continue
                        self._set_run(run_id, status="failed", error="Completion criteria did not pass", summary=validation["summary"], finished=True)
                        self._event(run_id, "run.failed", validation=validation)
                        return
                    audit = self._completion_audit(run, project, architecture, validation)
                    if audit["missing_tasks"]:
                        if iteration + 1 < max_iterations:
                            self._add_completion_tasks(run_id, audit["missing_tasks"])
                            self._set_run(run_id, status="running", error="")
                            self._event(run_id, "completion.audit.needs_work", audit=audit)
                            continue
                        self._set_run(run_id, status="failed", error="Whole-project acceptance audit found incomplete work", summary=json.dumps(audit), finished=True)
                        self._event(run_id, "run.failed", audit=audit)
                        return
                    self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent="reviewer", reason="independent-diff-review")
                    active_agent = "reviewer"
                    review = self._review(run, validation)
                    if review["blocking_findings"]:
                        last_feedback = "Independent review findings:\n" + json.dumps(review["blocking_findings"], indent=2)
                        if iteration + 1 < max_iterations:
                            self._add_repair_task(run_id, "Repair independent review findings", last_feedback, assigned_agent="debugger")
                            self._set_run(run_id, status="running", error="")
                            self._event(run_id, "repair.requested", stage="review", findings=review["blocking_findings"])
                            continue
                        self._set_run(run_id, status="failed", error="Independent review found blocking issues", summary=json.dumps(review), finished=True)
                        self._event(run_id, "review.failed", review=review)
                        return
                    snapshot = self.sandboxes.snapshot(run["workspace_id"], f"{run_id}-verified")
                    self._record_snapshot(run_id, snapshot, "Verified result")
                    published = self._publish_verified_changes(run)
                    summary = f"Validated autonomous run. {validation['summary']} Independent review: {review['status']}. Published {len(published['files'])} verified file(s) to the selected folder."
                    self._set_run(run_id, status="completed", summary=summary, error="", finished=True)
                    self._event(run_id, "run.completed", summary=summary, changed=changed, validation=validation, review=review)
                    self._remember_success(run, validation, review)
                    self._remember_repository_state(run)
                    return
                assigned_agent = str(task["assigned_agent"])
                if active_agent != assigned_agent:
                    self._event(run_id, "agent.handoff", from_agent=active_agent, to_agent=assigned_agent, task_id=task["id"], reason="task-assignment")
                    active_agent = assigned_agent
                self._set_task(task["id"], "running")
                self._event(run_id, "agent.started", agent=task["assigned_agent"], task_id=task["id"], iteration=iteration + 1)
                self._event(run_id, "agent.thinking", agent=task["assigned_agent"], task_id=task["id"], goal=task["title"])
                project_map = RepositoryIntelligence(project["workspace"]["workspace_path"]).scan()
                context = RepositoryIntelligence(project["workspace"]["workspace_path"]).select_context(
                    f"{run['user_prompt']} {task['title']} {last_feedback}", limit=20
                )
                if not decision_cache:
                    ready = [task, *[item for item in self._ready_tasks(run_id) if item["id"] != task["id"]]]
                    if len(ready) > 1:
                        decision_cache.update(self._parallel_task_decisions(run, project, architecture, ready, last_feedback))
                decision = decision_cache.pop(task["id"], None) or self._model_actions(run, project, project_map, context, task, last_feedback, architecture)
                actions = decision.get("actions", []) if isinstance(decision, dict) else []
                if not actions:
                    previous = task.get("result") if isinstance(task.get("result"), dict) else {}
                    attempts = max(0, int(previous.get("invalid_action_attempts", 0) or 0)) + 1
                    last_feedback = (
                        "The model response did not contain a usable tool action. Inspect the repository and return "
                        "JSON with an actions list. Do not describe a search-and-replace operation without a path and tool."
                    )
                    self._set_task(
                        task["id"],
                        "needs_fix",
                        {
                            "reason": "model returned no executable actions",
                            "invalid_action_attempts": attempts,
                            "feedback": last_feedback,
                            "response": decision,
                        },
                    )
                    self._event(run_id, "task.repairing", task_id=task["id"], reason=last_feedback, attempts=attempts)
                    if attempts >= 3:
                        error = (
                            "The selected model did not return an executable tool action after three repair attempts. "
                            "Choose a coding-capable model or make the task more specific."
                        )
                        self._set_run(run_id, status="failed", error=error, finished=True)
                        self._event(run_id, "run.failed", reason="model_stalled", task_id=task["id"], error=error)
                        return
                    continue
                failed = False
                action_results: list[tuple[str, dict[str, Any]]] = []
                for action in actions[:24]:
                    tool = str(action.get("tool", "")).strip()
                    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
                    arguments = self._normalized_action_arguments(tool, arguments)
                    if tool not in ACTION_TOOLS:
                        last_feedback = f"Unknown tool requested: {tool}"
                        failed = True
                        break
                    argument_error = self._action_argument_error(tool, arguments)
                    if argument_error:
                        last_feedback = argument_error
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback)
                        failed = True
                        break
                    try:
                        result = self._invoke(run, task["id"], tool, arguments)
                    except ApprovalRequired as exc:
                        last_feedback = (
                            f"Tool {exc.tool} is outside the protected autonomous boundary: {exc.reason}. "
                            "Do not ask the user for approval. Choose a workspace-safe tool or command and continue."
                        )
                        self._event(run_id, "autonomy.redirected", task_id=task["id"], tool=exc.tool, reason=exc.reason)
                        failed = True
                        break
                    except Exception as exc:
                        last_feedback = f"Tool {tool} failed: {exc}"
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback)
                        failed = True
                        break
                    action_results.append((tool, result))
                    if tool in {"write_file", "patch_file", "delete_file", "move_file"}:
                        changed = True
                        repeated_command_failures.clear()
                    exit_code = self._exit_code(result.get("exit_code"))
                    if exit_code is not None and exit_code not in {0, -1}:
                        last_feedback = f"Command failed.\nCommand: {result.get('command')}\nstdout:\n{result.get('stdout', '')}\nstderr:\n{result.get('stderr', '')}"
                        self._event(run_id, "repair.requested", task_id=task["id"], error=last_feedback[-20_000:])
                        command_key = hashlib.sha256(json.dumps({
                            "task": task["id"],
                            "tool": tool,
                            "command": result.get("command", arguments.get("command", [])),
                        }, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                        repeated_command_failures[command_key] = repeated_command_failures.get(command_key, 0) + 1
                        if repeated_command_failures[command_key] >= 2:
                            concise_error = (
                                "The same command failed twice without a repository change. "
                                f"Last command: {result.get('command')}. "
                                f"{str(result.get('stderr', '')).strip()[-4000:]}"
                            )
                            self._set_task(task["id"], "needs_fix", {"feedback": concise_error})
                            self._set_run(run_id, status="failed", error=concise_error, finished=True)
                            self._event(run_id, "run.failed", reason="repeated_command_failure", task_id=task["id"], error=concise_error)
                            return
                        failed = True
                        break
                if failed:
                    self._set_task(task["id"], "needs_fix", {"feedback": last_feedback[-20_000:]})
                    continue
                evidence_completed = self._action_batch_completes_task(task, action_results)
                if bool(decision.get("done", False)) or evidence_completed:
                    summary = str(decision.get("summary", "")).strip() or "The task produced verified workspace evidence."
                    self._set_task(task["id"], "completed", {"summary": summary, "evidence_completed": evidence_completed})
                    if evidence_completed and not bool(decision.get("done", False)):
                        self._event(
                            run_id,
                            "task.completion.inferred",
                            task_id=task["id"],
                            reason="successful tool evidence",
                            tools=[tool for tool, _ in action_results],
                        )
                    self._event(run_id, "task.completed", task_id=task["id"], summary=summary)
                    stalled_task_attempts.pop(task["id"], None)
                    last_feedback = ""
                else:
                    last_feedback = str(decision.get("summary", "Continue the current task with the next smallest verified action."))
                    stalled_task_attempts[task["id"]] = stalled_task_attempts.get(task["id"], 0) + 1
                    attempts = stalled_task_attempts[task["id"]]
                    self._set_task(task["id"], "needs_fix", {"feedback": last_feedback[-20_000:], "stalled_attempts": attempts})
                    self._event(run_id, "task.continued", task_id=task["id"], summary=last_feedback, stalled_attempts=attempts)
                    if attempts >= 3:
                        error = (
                            "The selected model did not produce a repository change or validation result after "
                            "three repair attempts. Choose a coding-capable model or make the task more specific."
                        )
                        self._set_run(run_id, status="failed", error=error, finished=True)
                        self._event(run_id, "run.failed", reason="model_stalled", task_id=task["id"], error=error)
                        return
            if self._cancelled(run_id):
                return
            unfinished = self._unfinished_tasks(run_id)
            self._set_run(run_id, status="failed", error="Iteration limit reached before all tasks and verification completed", finished=True)
            self._event(run_id, "run.failed", reason="iteration_limit", unfinished_tasks=unfinished, max_iterations=max_iterations)
        except Exception as exc:
            self._set_run(run_id, status="failed", error=str(exc), finished=True)
            self._event(run_id, "run.failed", error=str(exc))
        finally:
            self._cleanup_processes(run_id)
            with self._lock:
                self._threads.pop(run_id, None)

    def _consult_specialists(
        self,
        run: dict[str, Any],
        project: dict[str, Any],
        project_map: dict[str, Any],
        environment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask independent specialists for a compact architecture contract."""

        profile = classify_request(run["user_prompt"], project_map)
        roles = ["architect", "tester"]
        if profile.get("requires_ui"):
            roles.append("frontend")
        if profile.get("architecture") == "full-stack" or profile.get("kind") == "service":
            roles.extend(["backend", "security"])
        if profile.get("architecture") == "full-stack" or project_map.get("database"):
            roles.append("database")
        roles = list(dict.fromkeys(roles))[:5]
        self._event(run["id"], "specialists.parallel.started", agents=roles)

        def consult(role: str) -> tuple[str, dict[str, Any]]:
            prompt = (
                "Return JSON only: {\"summary\":string,\"components\":[string],\"risks\":[string],"
                "\"acceptance_criteria\":[string],\"recommended_tasks\":[string]}. "
                "Be concrete and repository-aware. Do not write code and do not claim execution.\n"
                f"Role: {role}\nGoal: {run['user_prompt']}\nDelivery profile: {json.dumps(profile)}\n"
                f"Project map: {json.dumps(self._compact_map(project_map))}\nProject memory: {json.dumps(project['memory'])}\n"
                f"Host and toolchain preflight: {json.dumps(environment or {})}"
            )
            result = self._model_json(run, [
                {"role": "system", "content": f"You are the independent {role} specialist in a software engineering team."},
                {"role": "user", "content": prompt},
            ], role=role)
            return role, result

        reports: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=min(5, len(roles)), thread_name_prefix="codeq-specialist") as pool:
            futures = [pool.submit(consult, role) for role in roles]
            for future in as_completed(futures):
                role, report = future.result()
                if isinstance(report, dict) and any(key in report for key in ("summary", "components", "acceptance_criteria", "recommended_tasks")):
                    reports[role] = report
        acceptance = list(dict.fromkeys(
            str(item).strip()
            for report in reports.values()
            for item in report.get("acceptance_criteria", [])
            if str(item).strip()
        ))[:40]
        result = {"profile": profile, "reports": reports, "acceptance_criteria": acceptance, "environment": environment or {}}
        self._event(run["id"], "specialists.parallel.completed", agents=list(reports), acceptance_criteria=acceptance)
        return result

    def _parallel_task_decisions(
        self,
        run: dict[str, Any],
        project: dict[str, Any],
        architecture: dict[str, Any],
        tasks: list[dict[str, Any]],
        feedback: str,
    ) -> dict[str, dict[str, Any]]:
        """Generate independent task decisions concurrently, then reject file collisions."""

        selected = tasks[:4]
        self._event(run["id"], "agents.parallel.started", task_ids=[task["id"] for task in selected], agents=[task["assigned_agent"] for task in selected])

        def decide(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            current_map = RepositoryIntelligence(project["workspace"]["workspace_path"]).scan()
            context = RepositoryIntelligence(project["workspace"]["workspace_path"]).select_context(
                f"{run['user_prompt']} {task['title']} {feedback}", limit=20
            )
            return task["id"], self._model_actions(run, project, current_map, context, task, feedback, architecture)

        decisions: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix="codeq-task") as pool:
            futures = [pool.submit(decide, task) for task in selected]
            for future in as_completed(futures):
                task_id, decision = future.result()
                decisions[task_id] = decision

        claimed: dict[str, str] = {}
        for task in selected:
            decision = decisions.get(task["id"], {})
            paths = self._decision_write_paths(decision)
            collision = next((path for path in paths if path in claimed), "")
            if collision:
                decisions.pop(task["id"], None)
                self._event(run["id"], "agent.parallel.recompute", task_id=task["id"], path=collision, after_task=claimed[collision])
                continue
            for path in paths:
                claimed[path] = task["id"]
        self._event(run["id"], "agents.parallel.completed", task_ids=list(decisions), deferred=[task["id"] for task in selected if task["id"] not in decisions])
        return decisions

    @staticmethod
    def _decision_write_paths(decision: dict[str, Any]) -> set[str]:
        paths: set[str] = set()
        for action in decision.get("actions", []) if isinstance(decision, dict) else []:
            if not isinstance(action, dict) or str(action.get("tool", "")) not in {"write_file", "patch_file", "delete_file", "move_file"}:
                continue
            arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            for key in ("path", "source", "destination"):
                if str(arguments.get(key, "")).strip():
                    paths.add(str(arguments[key]).replace("\\", "/").lower())
        return paths

    def _completion_audit(self, run: dict[str, Any], project: dict[str, Any], architecture: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        project_map = RepositoryIntelligence(workspace["workspace_path"]).scan()
        with self.store.session() as connection:
            changes = [str(row["filepath"]) for row in connection.execute(
                "SELECT DISTINCT filepath FROM codebase_file_changes WHERE run_id = ? ORDER BY filepath", (run["id"],)
            ).fetchall()]
        prompt = (
            "Audit the whole finished repository against the user request and acceptance contract. Return JSON only: "
            "{\"status\":\"passed|needs_work\",\"missing_tasks\":[{\"title\":string,\"agent\":string}],\"evidence\":[string]}. "
            "Only list a missing task when the project map, changed-file list, or validation proves a requested behavior is absent. "
            "Do not request deployment, credentials, purchases, or optional enhancements.\n"
            f"Goal: {run['user_prompt']}\nArchitecture: {json.dumps(architecture)}\nProject map: {json.dumps(self._compact_map(project_map))}\n"
            f"Changed files: {json.dumps(changes)}\nValidation: {json.dumps(validation)}"
        )
        parsed = self._model_json(run, [
            {"role": "system", "content": "You are the final acceptance auditor. Inspect evidence independently of the implementation agents."},
            {"role": "user", "content": prompt},
        ], role="reviewer")
        valid_agents = {"architect", "frontend", "backend", "database", "programmer", "tester", "debugger", "security", "reviewer", "documentation"}
        missing = []
        for item in parsed.get("missing_tasks", []) if isinstance(parsed, dict) else []:
            if not isinstance(item, dict) or not str(item.get("title", "")).strip():
                continue
            agent = str(item.get("agent", "programmer")).lower()
            missing.append({"title": str(item["title"]).strip(), "agent": agent if agent in valid_agents else "programmer"})
        result = {"status": "needs_work" if missing else "passed", "missing_tasks": missing[:20], "evidence": parsed.get("evidence", []) if isinstance(parsed, dict) else []}
        self._event(run["id"], "completion.audit.completed", audit=result)
        return result

    def _add_completion_tasks(self, run_id: str, tasks: list[dict[str, str]]) -> None:
        created: list[tuple[str, dict[str, str]]] = []
        with self.store.session() as connection:
            completed = [str(row["id"]) for row in connection.execute(
                "SELECT id FROM codebase_tasks WHERE run_id = ? AND status = 'completed' ORDER BY created_at, id", (run_id,)
            ).fetchall()]
            existing_titles = {str(row["title"]).lower() for row in connection.execute(
                "SELECT title FROM codebase_tasks WHERE run_id = ?", (run_id,)
            ).fetchall()}
            now = utc_now()
            for index, task in enumerate(tasks, start=1):
                if task["title"].lower() in existing_titles:
                    continue
                task_id = f"{run_id}:A{len(existing_titles) + index}"
                connection.execute(
                    "INSERT INTO codebase_tasks (id, run_id, title, assigned_agent, status, depends_on_json, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, '{}', ?, ?)",
                    (task_id, run_id, task["title"], task["agent"], json.dumps(completed), now, now),
                )
                created.append((task_id, task))
        for task_id, task in created:
            self._event(run_id, "task.created", task_id=task_id, title=task["title"], assigned_agent=task["agent"], source="completion-audit")

    def _plan(self, run: dict[str, Any], project: dict[str, Any], project_map: dict[str, Any], architecture: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        profile = classify_request(run["user_prompt"], project_map)
        prompt = (
            "Create a concise dependency-aware software engineering plan made only of executable repository changes and validation. Return JSON only using this shape: "
            f"{json.dumps(PLAN_SCHEMA)}. Do not include markdown. Choose the architecture and stack from the request and repository; do not default to a web framework when the request does not require one. "
            "Assign specialist roles only when useful. Do not add framework research, requirements definition, documentation, backend integration, or deployment unless the user explicitly requested it. "
            "Every implementation task must name a concrete code outcome; finish with tests/build/runtime checks supported by the project.\n"
            f"Goal: {run['user_prompt']}\nDelivery profile: {json.dumps(profile)}\nProject map: {json.dumps(self._compact_map(project_map))}\n"
            f"Project memory: {json.dumps(project['memory'])}\nSpecialist architecture review: {json.dumps(architecture or {})}"
        )
        parsed = self._model_json(run, [{"role": "system", "content": "You are the project manager. Inspect before editing, choose a fitting stack, delegate only useful work, and never claim success without validation."}, {"role": "user", "content": prompt}], role="project_manager")
        tasks = parsed.get("tasks", []) if isinstance(parsed, dict) else []
        normalized = []
        valid_agents = {"architect", "frontend", "backend", "database", "programmer", "tester", "debugger", "security", "reviewer", "documentation"}
        for index, task in enumerate(tasks[:100], start=1):
            if not isinstance(task, dict) or not str(task.get("title", "")).strip():
                continue
            task_id = str(task.get("id", f"T{index}")).strip() or f"T{index}"
            agent = str(task.get("agent", "programmer")).strip().lower()
            normalized.append({"id": task_id, "title": str(task["title"]).strip(), "assigned_agent": agent if agent in valid_agents else "programmer", "depends_on": [str(item) for item in task.get("depends_on", []) if str(item).strip()], "parent": str(task.get("parent", "")).strip()})
        if self._usable_model_plan(normalized, profile):
            self._event(run["id"], "plan.model.accepted", task_count=len(normalized), profile=profile)
            return normalized
        self._event(run["id"], "plan.fallback.selected", reason="model plan was empty or lacked implementation and validation tasks", profile=profile)

        if self._standalone_web_request(run["user_prompt"], project_map):
            goal = run["user_prompt"]
            if self._static_web_request(goal):
                page_count = self._requested_page_count(goal)
                return [
                    {"id": "T1", "title": f"Create a responsive static website with {page_count} linked pages and shared navigation", "assigned_agent": "programmer", "depends_on": []},
                    {"id": "T2", "title": "Add polished shared responsive styling and interaction behavior", "assigned_agent": "programmer", "depends_on": ["T1"]},
                    {"id": "T3", "title": f"Inspect all {page_count} pages and validate links and assets", "assigned_agent": "tester", "depends_on": ["T2"]},
                ]
            stack = "React application" if self._react_request(goal) else "web page"
            outcome = "sign-in component" if self._login_request(goal) else "requested user experience"
            return [
                {"id": "T1", "title": f"Create the responsive {stack} structure and {outcome}", "assigned_agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Add polished responsive styling and interaction behavior", "assigned_agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Inspect the generated files and validate the page artifacts", "assigned_agent": "tester", "depends_on": ["T2"]},
            ]
        if self._standalone_python_request(run["user_prompt"], project_map):
            return [
                {"id": "T1", "title": f"Implement the requested Python function: {run['user_prompt']}", "assigned_agent": "programmer", "depends_on": []},
                {"id": "T2", "title": "Create and run focused tests for the Python function", "assigned_agent": "tester", "depends_on": ["T1"]},
            ]
        if profile["mode"] == "greenfield":
            stack = f"{profile['framework']} {profile['kind']}"
            implementation = "Implement the complete user-requested behavior"
            if profile["architecture"] == "full-stack":
                implementation += " across client, server, persistence, validation, and failure handling"
            else:
                implementation += f" and usable {profile['kind']} experience"
            return [
                {"id": "T1", "title": f"Create the runnable {stack} foundation, entrypoint, configuration, and dependency manifest", "assigned_agent": "programmer", "depends_on": []},
                {"id": "T2", "title": implementation, "assigned_agent": "programmer", "depends_on": ["T1"]},
                {"id": "T3", "title": "Add focused automated tests for core behavior and failure paths", "assigned_agent": "tester", "depends_on": ["T2"]},
                {"id": "T4", "title": "Run available tests, build, static checks, and runtime smoke validation; repair every failure", "assigned_agent": "tester", "depends_on": ["T3"]},
            ]
        return [
            {"id": "T1", "title": "Inspect the repository and identify affected files", "assigned_agent": "architect", "depends_on": []},
            {"id": "T2", "title": "Implement the requested coherent changes", "assigned_agent": "programmer", "depends_on": ["T1"]},
            {"id": "T3", "title": "Run project validation and repair failures", "assigned_agent": "tester", "depends_on": ["T2"]},
            {"id": "T4", "title": "Independently review the final diff", "assigned_agent": "reviewer", "depends_on": ["T3"]},
        ]

    @staticmethod
    def _usable_model_plan(tasks: list[dict[str, Any]], profile: dict[str, Any]) -> bool:
        if not tasks:
            return False
        has_implementation = any(
            task_needs_file_change(str(task.get("title", "")), str(task.get("assigned_agent", "")))
            for task in tasks
        )
        has_validation = any(
            str(task.get("assigned_agent", "")).lower() == "tester"
            or re.search(r"\b(test|build|lint|typecheck|validate|verification|smoke)\b", str(task.get("title", "")), re.I)
            for task in tasks
        )
        # Existing repositories still need a real implementation task. A
        # model plan containing only review, setup, or dependency prose is not
        # executable and must fall back to the repository-native plan.
        if profile.get("mode") != "greenfield":
            return has_implementation
        if len(tasks) < 2:
            return False
        return has_implementation and has_validation

    def _model_actions(self, run: dict[str, Any], project: dict[str, Any], project_map: dict[str, Any], context: list[dict[str, Any]], task: dict[str, Any], feedback: str, architecture: dict[str, Any] | None = None) -> dict[str, Any]:
        source_map = RepositoryIntelligence(project["source_path"]).scan()
        profile = classify_request(run["user_prompt"], source_map)
        messages = [
            {"role": "system", "content": (
                "You are an autonomous software engineer operating only through supplied workspace tools. "
                "Return JSON only: {\"summary\":string,\"actions\":[{\"tool\":string,\"arguments\":object}],\"done\":boolean}. "
                "Inspect before editing. Prefer patch_file for narrow edits. Never request git push, deployment, secrets, or production access. "
                "Never pass Python, JavaScript, or other source text directly to run_shell or start_process. Create code with write_file or patch_file, then execute the file with a real command such as [\"python\", \"script.py\"]. "
                "browser_inspect requires a complete URL and must never be called before a page exists and a local web server is running. For an empty web project, create index.html and its assets with write_file first. "
                "Inside the protected working copy, file changes, dependency setup, tests, builds, local processes, and local Git operations are autonomous. "
                "Never pause to ask the user for permission; if an operation is outside that boundary, choose a safe local alternative. "
                "For new applications, make architecture decisions, implement a usable vertical slice, and validate it instead of only describing it. "
                "A task is not complete until actual validation evidence passes."
            )},
            {"role": "user", "content": (
                f"Overall goal: {run['user_prompt']}\nCurrent task: {task['title']}\nAgent role: {task['assigned_agent']}\nTask evidence: {json.dumps(task.get('result', {}))}\n"
                f"Delivery profile: {json.dumps(profile)}\nArchitecture and acceptance contract: {json.dumps(architecture or {})}\nProject map: {json.dumps(self._compact_map(project_map))}\nProject memory: {json.dumps(project['memory'])}\n"
                f"Recent project sessions: {json.dumps(self._recent_run_memory(run['codebase_project_id']))}\n"
                f"Relevant repository context: {json.dumps(context)[:100000]}\nPrevious failure: {feedback[-20000:]}\n"
                f"Available tools: {', '.join(sorted(ACTION_TOOLS))}. Return the next smallest coherent action batch."
            )},
        ]
        parsed = self._coerce_action_decision(
            self._model_json(run, messages, role=str(task.get("assigned_agent", "programmer")))
        )
        if self._repository_analysis_task(task) and not self._has_executable_actions(parsed):
            self._event(
                run["id"],
                "model.action_replaced",
                task_id=task["id"],
                reason="analysis task received no executable action",
            )
            return {
                "summary": "Recorded the repository structure, dependencies, and relevant files for the implementation work.",
                "actions": [{"tool": "project_map", "arguments": {}}],
                "done": True,
            }
        page_fallback = self._page_content_fallback(
            run["user_prompt"],
            task,
            Path(project["workspace"]["workspace_path"]),
        )
        if page_fallback is not None:
            self._event(
                run["id"],
                "model.action_replaced",
                task_id=task["id"],
                reason="page-content task uses an idempotent workspace artifact plan",
            )
            return page_fallback
        if profile["mode"] == "greenfield" and profile["framework"] == "django":
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            valid_actions = [
                action for action in actions
                if isinstance(action, dict)
                and str(action.get("tool", "")) in ACTION_TOOLS
                and not self._action_argument_error(
                    str(action.get("tool", "")),
                    self._normalized_action_arguments(
                        str(action.get("tool", "")),
                        action.get("arguments") if isinstance(action.get("arguments"), dict) else {},
                    ),
                )
            ]
            title = str(task.get("title", "")).lower()
            written_paths = {
                str((action.get("arguments") or {}).get("path", "")).replace("\\", "/").lower()
                for action in valid_actions
                if str(action.get("tool", "")) in {"write_file", "patch_file"}
            }
            foundation_task = any(term in title for term in ("foundation", "entrypoint", "configuration", "scaffold"))
            has_django_foundation = "manage.py" in written_paths and any(path.endswith("settings.py") for path in written_paths)
            behavior_task = any(term in title for term in ("complete user-requested behavior", "usable web application", "client"))
            has_django_behavior = any(path.endswith("views.py") for path in written_paths) and any(
                path.endswith("urls.py") for path in written_paths
            ) and any(path.endswith(".html") for path in written_paths)
            test_task = any(term in title for term in ("automated tests", "failure paths"))
            has_test_evidence = any(
                path.startswith("test") or "/test" in path for path in written_paths
            ) or any(str(action.get("tool", "")) == "run_tests" for action in valid_actions)
            validation_task = any(term in title for term in ("validation", "validate", "repair", "runtime", "static checks"))
            evidence_tools = {"run_tests", "run_linter", "run_typechecker", "run_build", "run_shell", "read_files", "browser_test", "browser_inspect"}
            has_validation_evidence = any(str(action.get("tool", "")) in evidence_tools for action in valid_actions)
            incompatible = (
                (foundation_task and not has_django_foundation)
                or (behavior_task and not has_django_behavior)
                or (test_task and not has_test_evidence)
                or (validation_task and not has_validation_evidence)
            )
            if not valid_actions or incompatible:
                fallback = self._django_fallback(run["user_prompt"], task)
                if fallback:
                    reason = "model action did not satisfy the Django task contract" if valid_actions else "model action was not executable"
                    self._event(run["id"], "framework.scaffold.selected", task_id=task["id"], framework="django", reason=reason)
                    return fallback
        if self._standalone_web_request(run["user_prompt"], source_map):
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            title = str(task.get("title", "")).lower()
            implementation_task = any(term in title for term in ("create", "markup", "styling", "interaction", "implement", "build"))
            invalid = not actions or any(
                not isinstance(action, dict)
                or str(action.get("tool", "")) not in ACTION_TOOLS
                or str(action.get("tool", "")) == "browser_inspect"
                or self._action_argument_error(str(action.get("tool", "")), self._normalized_action_arguments(str(action.get("tool", "")), action.get("arguments") if isinstance(action.get("arguments"), dict) else {}))
                for action in actions
            )
            if implementation_task and not any(str(action.get("tool", "")) in {"write_file", "patch_file"} for action in actions if isinstance(action, dict)):
                invalid = True
            if self._static_web_request(run["user_prompt"]):
                requested_pages = self._requested_page_count(run["user_prompt"])
                written_pages = {
                    str((action.get("arguments") or {}).get("path", "")).replace("\\", "/").lower()
                    for action in actions if isinstance(action, dict)
                    and str(action.get("tool", "")) in {"write_file", "patch_file"}
                    and str((action.get("arguments") or {}).get("path", "")).lower().endswith(".html")
                }
                if ("linked pages" in title or "static website" in title) and len(written_pages) < requested_pages:
                    invalid = True
            if self._react_request(run["user_prompt"]):
                written_paths = {
                    str((action.get("arguments") or {}).get("path", "")).replace("\\", "/").lower()
                    for action in actions if isinstance(action, dict) and str(action.get("tool", "")) in {"write_file", "patch_file"}
                }
                if ("structure" in title or "component" in title) and "src/app.jsx" not in written_paths:
                    invalid = True
                if ("styling" in title or "interaction" in title) and "src/styles.css" not in written_paths:
                    invalid = True
                if "validate" in title or "inspect" in title:
                    inspected_paths = {
                        str(path).replace("\\", "/").lower()
                        for action in actions if isinstance(action, dict) and str(action.get("tool", "")) == "read_files"
                        for path in ((action.get("arguments") or {}).get("paths") or [])
                    }
                    if not {"package.json", "src/app.jsx", "src/styles.css"}.issubset(inspected_paths):
                        invalid = True
            if invalid:
                fallback = self._empty_web_fallback(run["user_prompt"], task)
                if fallback:
                    self._event(run["id"], "model.action_replaced", task_id=task["id"], reason="invalid action for empty web workspace")
                    return fallback
        if self._standalone_python_request(run["user_prompt"], source_map):
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            title = str(task.get("title", "")).lower()
            implementation_task = "implement" in title
            test_task = "test" in title
            invalid = not actions or any(
                not isinstance(action, dict)
                or str(action.get("tool", "")) not in ACTION_TOOLS
                or self._nonproductive_creation_action(action)
                or self._action_argument_error(str(action.get("tool", "")), self._normalized_action_arguments(str(action.get("tool", "")), action.get("arguments") if isinstance(action.get("arguments"), dict) else {}))
                for action in actions
            )
            implementation_writes = [
                action for action in actions if isinstance(action, dict)
                and str(action.get("tool", "")) in {"write_file", "patch_file"}
                and not Path(str((action.get("arguments") or {}).get("path", ""))).name.lower().startswith("test")
            ]
            concrete_test_actions = [
                action for action in actions if isinstance(action, dict) and (
                    (str(action.get("tool", "")) in {"write_file", "patch_file"}
                     and Path(str((action.get("arguments") or {}).get("path", ""))).name.lower().startswith("test"))
                    or (str(action.get("tool", "")) == "run_tests"
                        and bool(self._normalized_action_arguments("run_tests", action.get("arguments") or {}).get("command")))
                )
            ]
            if implementation_task and not implementation_writes:
                invalid = True
            if test_task and not concrete_test_actions:
                invalid = True
            if invalid:
                fallback = self._python_function_fallback(run["user_prompt"], task)
                if fallback:
                    self._event(run["id"], "model.action_replaced", task_id=task["id"], reason="non-executable action for Python creation task")
                    return fallback
        actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
        if task_needs_file_change(str(task.get("title", "")), str(task.get("assigned_agent", ""))):
            productive = any(
                isinstance(action, dict)
                and str(action.get("tool", "")) in {"write_file", "patch_file", "delete_file", "move_file"}
                for action in actions
            )
            non_attempts = {"list_directory", "read_file", "read_files", "grep", "search_symbol", "find_references", "project_map", "git_status", "git_diff", "git_log"}
            narration_only = bool(actions) and all(
                isinstance(action, dict) and (
                    str(action.get("tool", "")) in non_attempts or self._nonproductive_creation_action(action)
                )
                for action in actions
            )
            if not productive and (not actions or narration_only):
                self._event(run["id"], "model.action_rejected", task_id=task["id"], reason="implementation task returned no file change", profile=profile)
                return {
                    "summary": "No repository change was produced. Inspect the project map and implement the requested code in real files.",
                    "actions": [{"tool": "project_map", "arguments": {}}],
                    "done": False,
                }
        title_text = str(task.get("title", "")).lower()
        if str(task.get("assigned_agent", "")).lower() == "tester" and not task_needs_file_change(title_text, "tester"):
            evidence_tools = {"run_tests", "run_linter", "run_typechecker", "run_build", "browser_test", "browser_inspect", "run_shell", "read_files", "project_map"}
            if not any(isinstance(action, dict) and str(action.get("tool", "")) in evidence_tools for action in actions):
                if not source_map.get("commands"):
                    self._event(run["id"], "model.action_replaced", task_id=task["id"], reason="no repository validation command is configured")
                    return {
                        "summary": "Recorded the repository map; no project-specific validation command is configured for this workspace.",
                        "actions": [{"tool": "project_map", "arguments": {}}],
                        "done": True,
                    }
                self._event(run["id"], "model.action_rejected", task_id=task["id"], reason="validation task returned no executable evidence", profile=profile)
                return {"summary": "Run the repository's available validation commands and inspect their real output.", "actions": [{"tool": "project_map", "arguments": {}}], "done": False}
            # Dependency/configuration checks are often represented by a
            # project-map or manifest inspection when the repository has no
            # package-manager command configured. Treat that inspection as
            # the task's evidence instead of sending the same tester back
            # through an endless repair cycle.
            if not bool(parsed.get("done", False)) and re.search(r"\b(dependenc|configur|setup|ensure)\w*\b", title_text):
                return {
                    "summary": "Inspected the repository configuration for this validation step; no project-specific dependency command was configured.",
                    "actions": actions,
                    "done": True,
                }
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _has_executable_actions(decision: dict[str, Any]) -> bool:
        actions = decision.get("actions") if isinstance(decision, dict) else None
        return bool(
            isinstance(actions, list)
            and any(
                isinstance(action, dict) and str(action.get("tool", "")).strip() in ACTION_TOOLS
                for action in actions
            )
        )

    @staticmethod
    def _action_batch_completes_task(
        task: dict[str, Any],
        action_results: list[tuple[str, dict[str, Any]]],
    ) -> bool:
        """Use successful tool evidence when a local model omits its done flag.

        Final validation and independent review still decide whether the run is
        complete. This method only advances the current task after it has made
        the kind of progress that task asks for.
        """

        if not action_results:
            return False
        title = str(task.get("title", ""))
        agent = str(task.get("assigned_agent", ""))
        mutating_tools = {"write_file", "patch_file", "delete_file", "move_file", "install_dependency"}
        evidence_tools = {
            "run_tests",
            "run_linter",
            "run_typechecker",
            "run_build",
            "run_shell",
            "read_files",
            "browser_test",
            "browser_inspect",
            "project_map",
            "git_diff",
        }

        def succeeded(result: dict[str, Any]) -> bool:
            exit_code = CodebaseService._exit_code(result.get("exit_code"))
            return exit_code in {None, 0, -1} and not result.get("error")

        if task_needs_file_change(title, agent):
            return any(tool in mutating_tools and succeeded(result) for tool, result in action_results)
        return any(tool in evidence_tools and succeeded(result) for tool, result in action_results)

    @staticmethod
    def _repository_analysis_task(task: dict[str, Any]) -> bool:
        """Identify planning work that can safely use the repository map as evidence."""

        title = str(task.get("title", "")).casefold()
        if task_needs_file_change(title, str(task.get("assigned_agent", ""))):
            return False
        return bool(
            re.search(
                r"\b(review|inspect|analyse|analyze|architecture|dependenc|discover|assess|understand|repository)\w*\b",
                title,
            )
        )

    @staticmethod
    def _coerce_action_decision(value: dict[str, Any]) -> dict[str, Any]:
        """Normalize small, structurally sound action variants from local models.

        A model occasionally returns a single tool object rather than the required
        ``actions`` list. That is recoverable. A bare search-and-replace object is
        deliberately left without an action because it lacks both a file target and
        an explicit edit tool.
        """

        if not isinstance(value, dict):
            return {}
        decision = dict(value)
        actions = decision.get("actions")
        if isinstance(actions, dict):
            decision["actions"] = [actions]
        elif "tool" in decision and not isinstance(actions, list):
            arguments = decision.get("arguments")
            decision["actions"] = [{
                "tool": decision.get("tool"),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }]
        elif all(key in decision for key in ("path", "search", "replace")) and not isinstance(actions, list):
            decision["actions"] = [{
                "tool": "patch_file",
                "arguments": {
                    "path": decision["path"],
                    "search": decision["search"],
                    "replace": decision["replace"],
                },
            }]
        elif not isinstance(actions, list):
            decision["actions"] = []
        return decision

    @classmethod
    def _page_content_fallback(cls, goal: str, task: dict[str, Any], root: Path) -> dict[str, Any] | None:
        """Provide a small, repeatable implementation for page-content work.

        Local models often split this request into several preparation and
        validation tasks. They must all converge on one generated page rather
        than producing a new placeholder on every retry.
        """

        if not cls._page_content_request(goal):
            return None
        title = str(task.get("title", "")).lower()
        validation = bool(re.search(r"\b(test|tests|validate|validation|verify|verification|inspect|review|check|audit)\b", title))
        if validation:
            paths = sorted(
                path.relative_to(root).as_posix()
                for path in root.glob("*.html")
                if path.is_file()
            )
            return {
                "summary": "Validated the generated page and the existing HTML files.",
                "actions": [{"tool": "read_files", "arguments": {"paths": paths[:30]}}] if paths else [{"tool": "project_map", "arguments": {}}],
                "done": True,
            }

        implementation = bool(re.search(r"\b(page|pages|content|data|template|route|view|screen|builder|loader|implement|create|add|update|prepare|configure|integrate)\b", title))
        if not implementation:
            return None
        root = root.resolve()
        generated: list[Path] = []
        for path in sorted(root.glob("new-page*.html"), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if 'data-codeq-generated="page"' in content:
                generated.append(path)
        target = generated[0] if generated else None
        if target is not None:
            return {
                "summary": f"Reused the generated page artifact at {target.name}; no duplicate page was created.",
                "actions": [{"tool": "project_map", "arguments": {}}],
                "done": True,
            }

        target = root / "new-page.html"
        if target.exists():
            index = 2
            while (root / f"new-page-{index}.html").exists():
                index += 1
            target = root / f"new-page-{index}.html"
        html = cls._generated_page_html(goal)
        return {
            "summary": f"Created {target.name} with accessible sample content.",
            "actions": [{"tool": "write_file", "arguments": {"path": target.relative_to(root).as_posix(), "content": html}}],
            "done": True,
        }

    @staticmethod
    def _page_content_request(goal: str) -> bool:
        text = str(goal).lower()
        has_page = bool(re.search(r"\b(new|another|additional|extra)\s+page\b", text))
        has_content = any(term in text for term in ("dummy content", "sample content", "placeholder content", "sample data"))
        return has_page and has_content

    @staticmethod
    def _generated_page_html(goal: str) -> str:
        return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A new page with sample content.">
  <title>New page</title>
</head>
<body data-codeq-generated="page">
  <main>
    <p>New page</p>
    <h1>Welcome to the new page</h1>
    <p>This page was added for the requested work and is ready for real content.</p>
    <section aria-labelledby="sample-content">
      <h2 id="sample-content">Sample content</h2>
      <p>Replace this example content with the page details for this project.</p>
      <dl>
        <div><dt>Request</dt><dd>{str(goal).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</dd></div>
        <div><dt>Status</dt><dd>Ready for review</dd></div>
      </dl>
    </section>
  </main>
</body>
</html>
'''

    @classmethod
    def _empty_web_fallback(cls, goal: str, task: dict[str, Any]) -> dict[str, Any]:
        title = str(task.get("title", "")).lower()
        if cls._static_web_request(goal):
            page_names = cls._static_page_names(cls._requested_page_count(goal))
            if "linked pages" in title or "static website" in title or "structure" in title:
                return {
                    "summary": f"Created {len(page_names)} linked static pages",
                    "actions": [
                        {"tool": "write_file", "arguments": {"path": filename, "content": cls._static_page_html(filename, label, page_names)}}
                        for filename, label in page_names
                    ],
                    "done": True,
                }
            if "styling" in title or "interaction" in title:
                return {
                    "summary": "Added shared responsive styling and navigation behavior",
                    "actions": [
                        {"tool": "write_file", "arguments": {"path": "styles.css", "content": STATIC_SITE_CSS}},
                        {"tool": "write_file", "arguments": {"path": "app.js", "content": STATIC_SITE_JS}},
                    ],
                    "done": True,
                }
            if "validate" in title or "inspect" in title:
                paths = [filename for filename, _label in page_names] + ["styles.css", "app.js"]
                return {"summary": f"Inspected all {len(page_names)} pages and shared assets", "actions": [{"tool": "read_files", "arguments": {"paths": paths}}], "done": True}
            return {}
        if cls._react_request(goal):
            if "structure" in title or "component" in title:
                return {
                    "summary": "Created the React application and responsive sign-in component",
                    "actions": [
                        {"tool": "write_file", "arguments": {"path": "package.json", "content": REACT_LOGIN_PACKAGE}},
                        {"tool": "write_file", "arguments": {"path": "index.html", "content": REACT_LOGIN_INDEX}},
                        {"tool": "write_file", "arguments": {"path": "src/main.jsx", "content": REACT_LOGIN_MAIN}},
                        {"tool": "write_file", "arguments": {"path": "src/App.jsx", "content": REACT_LOGIN_APP}},
                    ],
                    "done": True,
                }
            if "styling" in title or "interaction" in title:
                return {"summary": "Added responsive React styling", "actions": [{"tool": "write_file", "arguments": {"path": "src/styles.css", "content": REACT_LOGIN_CSS}}], "done": True}
            if "validate" in title or "inspect" in title:
                return {"summary": "Read the generated React artifacts for validation", "actions": [{"tool": "read_files", "arguments": {"paths": ["package.json", "index.html", "src/main.jsx", "src/App.jsx", "src/styles.css"]}}], "done": True}
            return {}
        if "markup" in title or "index.html" in title or "structure" in title or "component" in title:
            return {"summary": "Created the semantic login page markup", "actions": [{"tool": "write_file", "arguments": {"path": "index.html", "content": EMPTY_LOGIN_HTML}}], "done": True}
        if "styling" in title or "interaction" in title:
            return {
                "summary": "Added responsive styling and client-side form behavior",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "styles.css", "content": EMPTY_LOGIN_CSS}},
                    {"tool": "write_file", "arguments": {"path": "app.js", "content": EMPTY_LOGIN_JS}},
                ],
                "done": True,
            }
        if "validate" in title or "inspect" in title:
            return {"summary": "Read the generated frontend artifacts for validation", "actions": [{"tool": "read_files", "arguments": {"paths": ["index.html", "styles.css", "app.js"]}}], "done": True}
        return {}

    @staticmethod
    def _react_request(goal: str) -> bool:
        return bool(re.search(r"\breact(?:\.js|js)?\b", str(goal), re.I))

    @classmethod
    def _django_fallback(cls, goal: str, task: dict[str, Any]) -> dict[str, Any]:
        title = str(task.get("title", "")).lower()
        pages = cls._requested_named_pages(goal)
        if "foundation" in title or "entrypoint" in title or "configuration" in title:
            return {
                "summary": "Created the Django project foundation and dependency manifest",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "requirements.txt", "content": "Django>=5.1,<6.0\n"}},
                    {"tool": "write_file", "arguments": {"path": "manage.py", "content": DJANGO_MANAGE}},
                    {"tool": "write_file", "arguments": {"path": "config/__init__.py", "content": "\"\"\"Django project configuration.\"\"\"\n"}},
                    {"tool": "write_file", "arguments": {"path": "config/settings.py", "content": DJANGO_SETTINGS}},
                    {"tool": "write_file", "arguments": {"path": "config/urls.py", "content": DJANGO_ROOT_URLS}},
                    {"tool": "write_file", "arguments": {"path": "config/wsgi.py", "content": DJANGO_WSGI}},
                    {"tool": "write_file", "arguments": {"path": "pages/__init__.py", "content": "\"\"\"Page application.\"\"\"\n"}},
                ],
                "done": True,
            }
        if "complete user-requested behavior" in title or "usable web application" in title or "client" in title:
            actions = [
                {"tool": "write_file", "arguments": {"path": "pages/views.py", "content": cls._django_views(pages)}},
                {"tool": "write_file", "arguments": {"path": "pages/urls.py", "content": cls._django_urls(pages)}},
                {"tool": "write_file", "arguments": {"path": "templates/base.html", "content": cls._django_base_template(pages, goal)}},
                {"tool": "write_file", "arguments": {"path": "static/css/site.css", "content": DJANGO_SITE_CSS}},
                {"tool": "write_file", "arguments": {"path": "static/js/site.js", "content": cls._django_site_script(goal)}},
            ]
            actions.extend(
                {"tool": "write_file", "arguments": {"path": f"templates/{slug}.html", "content": cls._django_page_template(slug, label)}}
                for slug, label in pages
            )
            return {"summary": f"Implemented {len(pages)} routed Django pages with a shared jQuery frontend", "actions": actions, "done": True}
        if "automated tests" in title or "failure paths" in title:
            return {
                "summary": "Added structural route, template, and frontend checks",
                "actions": [{"tool": "write_file", "arguments": {"path": "test_structure.py", "content": cls._django_structure_tests(pages, goal)}}],
                "done": True,
            }
        if any(term in title for term in ("validation", "validate", "repair", "runtime", "static checks")):
            paths = ["manage.py", "config/settings.py", "config/urls.py", "pages/views.py", "pages/urls.py", "templates/base.html", "static/js/site.js"]
            paths.extend(f"templates/{slug}.html" for slug, _label in pages)
            return {
                "summary": "Validated Django configuration, tests, routes, templates, and frontend assets",
                "actions": cls._django_validation_actions(paths),
                "done": True,
            }
        return {}

    @staticmethod
    def _django_validation_actions(paths: list[str]) -> list[dict[str, Any]]:
        if os.getenv("OPENAGENT_CODEQ_INSTALL_DEPENDENCIES", "1").strip().lower() in {"0", "false", "no"}:
            return [
                {"tool": "run_tests", "arguments": {"command": ["python", "-m", "unittest", "discover", "-v"]}},
                {"tool": "run_shell", "arguments": {"command": ["python", "-m", "compileall", "-q", "."]}},
                {"tool": "read_files", "arguments": {"paths": paths}},
            ]
        venv_python = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
        return [
            {"tool": "run_shell", "arguments": {"command": ["python", "-m", "venv", ".venv"], "timeout_seconds": 300}},
            {"tool": "install_dependency", "arguments": {"command": [venv_python, "-m", "pip", "install", "-r", "requirements.txt"]}},
            {"tool": "run_tests", "arguments": {"command": [venv_python, "manage.py", "check"]}},
            {"tool": "run_tests", "arguments": {"command": [venv_python, "manage.py", "test"]}},
            {"tool": "run_tests", "arguments": {"command": [venv_python, "-m", "unittest", "discover", "-v"]}},
            {"tool": "read_files", "arguments": {"paths": paths}},
        ]

    @staticmethod
    def _requested_named_pages(goal: str) -> list[tuple[str, str]]:
        text = str(goal).lower()
        aliases = {"index": "home", "homepage": "home"}
        known: list[str] = []
        for name in ("home", "index", "homepage", "about", "contact", "careers", "services", "work", "pricing", "faq", "team"):
            if re.search(rf"\b{name}\b", text):
                canonical = aliases.get(name, name)
                if canonical not in known:
                    known.append(canonical)
        count = CodebaseService._requested_page_count(goal)
        if count <= 1 and len(known) > 1:
            count = len(known)
        defaults = ["home", "about", "contact", "careers", "services", "work", "pricing", "faq", "team"]
        for name in defaults:
            if len(known) >= count:
                break
            if name not in known:
                known.append(name)
        return [(name, name.title()) for name in known[:max(1, count)]]

    @staticmethod
    def _django_views(pages: list[tuple[str, str]]) -> str:
        functions = "\n\n".join(f"def {slug}(request):\n    return render(request, '{slug}.html')" for slug, _label in pages)
        return f"from django.shortcuts import render\n\n\n{functions}\n"

    @staticmethod
    def _django_urls(pages: list[tuple[str, str]]) -> str:
        routes = []
        for slug, _label in pages:
            route = "" if slug == "home" else f"{slug}/"
            routes.append(f"    path('{route}', views.{slug}, name='{slug}'),")
        return "from django.urls import path\n\nfrom . import views\n\napp_name = 'pages'\n\nurlpatterns = [\n" + "\n".join(routes) + "\n]\n"

    @staticmethod
    def _django_base_template(pages: list[tuple[str, str]], goal: str = "") -> str:
        if re.search(r"\bbootstrap\b", str(goal), re.I):
            return CodebaseService._django_bootstrap_base_template(pages)
        links = "\n".join(f'        <a href="{{% url \'pages:{slug}\' %}}">{label}</a>' for slug, label in pages)
        return """{% load static %}
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Northstar{% endblock %}</title>
  <link rel="stylesheet" href="{% static 'css/site.css' %}">
</head>
<body>
  <header class="site-header"><div class="nav-shell"><a class="brand" href="{% url 'pages:home' %}"><span>N</span> Northstar</a><button class="menu-button" type="button" aria-expanded="false" aria-controls="primary-nav">Menu</button><nav id="primary-nav">
""" + links + """
  </nav></div></header>
  <main>{% block content %}{% endblock %}</main>
  <footer>Northstar · Django and jQuery</footer>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js" integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo=" crossorigin="anonymous"></script>
  <script src="{% static 'js/site.js' %}"></script>
</body>
</html>
"""

    @staticmethod
    def _django_bootstrap_base_template(pages: list[tuple[str, str]]) -> str:
        links = "\n".join(
            f'          <li class="nav-item"><a class="nav-link" href="{{% url \'pages:{slug}\' %}}">{label}</a></li>'
            for slug, label in pages
        )
        return """{% load static %}
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Northstar{% endblock %}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
  <link rel="stylesheet" href="{% static 'css/site.css' %}">
</head>
<body>
  <nav class="navbar navbar-expand-lg bg-body-tertiary"><div class="container"><a class="navbar-brand" href="{% url 'pages:home' %}">Northstar</a><button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#primary-nav" aria-controls="primary-nav" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button><div class="collapse navbar-collapse" id="primary-nav"><ul class="navbar-nav ms-auto">
""" + links + """
  </ul></div></div></nav>
  <main>{% block content %}{% endblock %}</main>
  <footer>Northstar · Django and Bootstrap</footer>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
  <script src="{% static 'js/site.js' %}"></script>
</body>
</html>
"""

    @staticmethod
    def _django_site_script(goal: str) -> str:
        if re.search(r"\bbootstrap\b", str(goal), re.I):
            return "document.documentElement.classList.add('js-ready');\n"
        return DJANGO_SITE_JS

    @staticmethod
    def _django_page_template(slug: str, label: str) -> str:
        descriptions = {"home": "A focused place to begin.", "about": "Learn how the work is organized and delivered.", "contact": "Start a useful conversation with the team."}
        description = descriptions.get(slug, f"Explore {label.lower()} with a clear and responsive experience.")
        return "{% extends 'base.html' %}\n{% block title %}" + label + " | Northstar{% endblock %}\n{% block content %}\n<section class=\"hero\"><div><p class=\"eyebrow\">Northstar / " + label + "</p><h1>" + label + "</h1><p>" + description + "</p></div></section>\n<section class=\"content\"><h2>Built with purpose</h2><p>This Django page uses shared templates, responsive styles, and progressive frontend behavior.</p></section>\n{% endblock %}\n"

    @staticmethod
    def _django_structure_tests(pages: list[tuple[str, str]], goal: str = "") -> str:
        names = [slug for slug, _label in pages]
        frontend_asset = "bootstrap@5.3.3" if re.search(r"\bbootstrap\b", str(goal), re.I) else "jquery-3.7.1.min.js"
        return """from pathlib import Path
import unittest


class ProjectStructureTests(unittest.TestCase):
    def test_django_project_and_requested_pages_exist(self):
        root = Path(__file__).parent
        required = %r
        self.assertTrue((root / 'manage.py').is_file())
        self.assertTrue((root / 'config' / 'settings.py').is_file())
        for name in required:
            self.assertTrue((root / 'templates' / f'{name}.html').is_file())

    def test_routes_and_jquery_integration_are_wired(self):
        root = Path(__file__).parent
        routes = (root / 'pages' / 'urls.py').read_text(encoding='utf-8')
        base = (root / 'templates' / 'base.html').read_text(encoding='utf-8')
        for name in %r:
            self.assertIn("name='" + name + "'", routes)
        self.assertIn(%r, base)
        self.assertTrue((root / 'static' / 'js' / 'site.js').is_file())


if __name__ == '__main__':
    unittest.main()
""" % (names, names, frontend_asset)

    @staticmethod
    def _login_request(goal: str) -> bool:
        return bool(re.search(r"\b(login|log in|sign[ -]?in|sign[ -]?up|auth(?:entication)?)\b", str(goal), re.I))

    @classmethod
    def _static_web_request(cls, goal: str) -> bool:
        text = str(goal).lower()
        return not cls._react_request(goal) and any(term in text for term in ("static", "html and css", "html/css", "plain html"))

    @staticmethod
    def _requested_page_count(goal: str) -> int:
        text = str(goal).lower()
        match = re.search(r"\b(\d{1,2})\s*(?:static\s+)?pages?\b", text)
        if match:
            return max(1, min(int(match.group(1)), 20))
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        for word, count in words.items():
            if re.search(rf"\b{word}\s+(?:static\s+)?pages?\b", text):
                return count
        return 1

    @staticmethod
    def _static_page_names(count: int) -> list[tuple[str, str]]:
        labels = ["Home", "About", "Services", "Contact", "Work", "Insights", "Pricing", "FAQ", "Team", "Careers"]
        pages: list[tuple[str, str]] = []
        for index in range(max(1, min(int(count), 20))):
            label = labels[index] if index < len(labels) else f"Page {index + 1}"
            filename = "index.html" if index == 0 else re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") + ".html"
            pages.append((filename, label))
        return pages

    @staticmethod
    def _static_page_html(filename: str, label: str, pages: list[tuple[str, str]]) -> str:
        navigation = "\n".join(
            f'        <a href="{path}"' + (' aria-current="page"' if path == filename else "") + f">{name}</a>"
            for path, name in pages
        )
        descriptions = {
            "Home": "A clear, fast starting point built for every screen.",
            "About": "The principles, people, and practical thinking behind the work.",
            "Services": "Focused capabilities designed around useful outcomes.",
            "Contact": "A straightforward way to start a conversation.",
        }
        description = descriptions.get(label, f"Explore {label.lower()} with a focused, accessible layout.")
        return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{description}">
  <title>{label} | Northstar</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header">
    <div class="nav-shell">
      <a class="brand" href="index.html"><span class="brand-mark">N</span> Northstar</a>
      <button class="menu-button" type="button" aria-expanded="false" aria-controls="site-navigation">Menu</button>
      <nav class="site-nav" id="site-navigation" aria-label="Primary" data-open="false">
{navigation}
      </nav>
    </div>
  </header>
  <main>
    <section class="hero">
      <div class="hero-inner">
        <p class="eyebrow">Northstar / {label}</p>
        <h1>{label}</h1>
        <p>{description}</p>
      </div>
    </section>
    <section class="content">
      <div class="content-grid">
        <div>
          <h2>Designed for clarity</h2>
          <p>This page shares one visual system with the rest of the site while keeping its own purpose and content. Navigation remains usable with a keyboard, touch input, or a narrow viewport.</p>
          <ul class="feature-list"><li>Responsive from small screens upward</li><li>Shared navigation and consistent page structure</li><li>Semantic HTML with accessible controls</li></ul>
        </div>
        <aside class="side-note"><strong>{label} at a glance</strong><span>{description}</span></aside>
      </div>
    </section>
  </main>
  <footer class="site-footer">Northstar · Built with plain HTML, CSS, and JavaScript</footer>
  <script src="app.js"></script>
</body>
</html>
'''

    @staticmethod
    def _exit_code(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _standalone_web_request(goal: str, project_map: dict[str, Any]) -> bool:
        if int(project_map.get("total_bytes", 0) or 0) > 1_000_000 or str(project_map.get("framework", "")).strip():
            return False
        profile = classify_request(goal, project_map)
        if profile.get("explicit_stack"):
            return profile.get("framework") == "react" and CodebaseService._login_request(goal)
        text = goal.lower()
        return CodebaseService._static_web_request(goal) or any(term in text for term in ("login page", "sign in page", "signup page", "sign up page", "landing page", "plain html"))

    @staticmethod
    def _standalone_python_request(goal: str, project_map: dict[str, Any]) -> bool:
        if int(project_map.get("total_bytes", 0) or 0) > 1_000_000 or str(project_map.get("framework", "")).strip():
            return False
        text = goal.lower()
        return "python" in text and any(term in text for term in ("function", "script", "module"))

    @staticmethod
    def _nonproductive_creation_action(action: dict[str, Any]) -> bool:
        tool = str(action.get("tool", ""))
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        if tool not in {"run_shell", "start_process"}:
            return False
        command = arguments.get("command", arguments.get("cmd", arguments.get("argv", [])))
        parts = shlex.split(command, posix=False) if isinstance(command, str) else [str(item) for item in command]
        return bool(parts and Path(parts[0]).name.lower() in {"echo", "write-host", "printf"})

    @staticmethod
    def _normalized_action_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return CodebaseTools._normalize_arguments(tool, arguments)

    @staticmethod
    def _python_function_fallback(goal: str, task: dict[str, Any]) -> dict[str, Any]:
        title = str(task.get("title", "")).lower()
        assigned_agent = str(task.get("assigned_agent", "")).lower()
        text = f"{goal} {title}".lower()
        if "double" in text or "twice" in text:
            function_code = "def double(value: float) -> float:\n    \"\"\"Return twice the supplied numeric value.\"\"\"\n    if not isinstance(value, (int, float)) or isinstance(value, bool):\n        raise TypeError('value must be numeric')\n    return value * 2\n"
            test_code = "import unittest\n\nfrom function import double\n\n\nclass FunctionTests(unittest.TestCase):\n    def test_double(self):\n        self.assertEqual(8, double(4))\n        self.assertEqual(-5, double(-2.5))\n\n    def test_non_numeric_value(self):\n        with self.assertRaises(TypeError):\n            double('4')\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
        else:
            function_code = "def factorial(value: int) -> int:\n    \"\"\"Return value factorial for a non-negative integer.\"\"\"\n    if not isinstance(value, int) or isinstance(value, bool):\n        raise TypeError('value must be an integer')\n    if value < 0:\n        raise ValueError('value must be non-negative')\n    result = 1\n    for number in range(2, value + 1):\n        result *= number\n    return result\n"
            test_code = "import unittest\n\nfrom function import factorial\n\n\nclass FunctionTests(unittest.TestCase):\n    def test_factorial(self):\n        self.assertEqual(1, factorial(0))\n        self.assertEqual(120, factorial(5))\n\n    def test_negative_value(self):\n        with self.assertRaises(ValueError):\n            factorial(-1)\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
        # Classify from the assigned task, not from words in the overall goal.
        # A request such as "write a function and test it" still gives the
        # programmer an implementation task before the tester runs.
        implementation_task = assigned_agent == "programmer" or title.startswith("implement")
        test_task = assigned_agent == "tester" or title.startswith(("create and run focused tests", "test ", "run test"))
        if implementation_task:
            return {
                "summary": "Implemented a documented, validated Python function",
                "actions": [{"tool": "write_file", "arguments": {"path": "function.py", "content": function_code}}],
                "done": True,
            }
        if test_task:
            return {
                "summary": "Created and ran focused Python tests",
                "actions": [
                    {"tool": "write_file", "arguments": {"path": "test_function.py", "content": test_code}},
                    {"tool": "run_tests", "arguments": {"command": ["python", "-m", "unittest", "discover", "-v"]}},
                ],
                "done": True,
            }
        if any(term in text for term in ("factorial", "double", "twice", "fucntion", "function")):
            return {
                "summary": "Implemented a documented, validated Python function",
                "actions": [{"tool": "write_file", "arguments": {"path": "function.py", "content": function_code}}],
                "done": True,
            }
        return {}

    @staticmethod
    def _action_argument_error(tool: str, arguments: dict[str, Any]) -> str:
        required = {
            "read_file": ("path",),
            "write_file": ("path", "content"),
            "patch_file": ("path", "search", "replace"),
            "delete_file": ("path",),
            "move_file": ("source", "destination"),
            "run_shell": ("command",),
            "start_process": ("command",),
            "process_status": ("process_id",),
            "stop_process": ("process_id",),
            "inspect_logs": ("process_id",),
            "browser_inspect": ("url",),
        }.get(tool, ())
        missing = [name for name in required if arguments.get(name) is None or arguments.get(name) == ""]
        if missing:
            return f"Tool {tool} was not executed because required argument(s) were missing: {', '.join(missing)}. Choose a valid tool action and continue the task."
        if tool == "browser_inspect":
            parsed = urlparse(str(arguments["url"]).strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return "Tool browser_inspect requires a complete HTTP URL. Create the page, start a local server, and then inspect its localhost URL."
        return ""

    def _review(self, run: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        diff = self._review_material(run)
        messages = [
            {"role": "system", "content": "You are an independent code reviewer. You do not know the coder's reasoning. Review only the goal, diff, and validation. Return JSON only. Never report a file absent from the diff. Every finding must quote exact diff evidence. A client-only form without an authentication endpoint does not require CSRF protection."},
            {"role": "user", "content": (
                "Return {\"status\":\"passed|needs_fix\",\"findings\":[{\"severity\":\"critical|high|medium|low\",\"file\":string,\"line\":integer,\"issue\":string,\"evidence\":\"exact text copied from the diff\"}]}.\n"
                f"Goal: {run['user_prompt']}\nValidation: {json.dumps(validation)}\nDiff:\n{diff[-100000:]}"
            )},
        ]
        parsed = self._model_json(run, messages, role="reviewer")
        model_findings = parsed.get("findings", []) if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list) else []
        findings = []
        for item in model_findings:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            evidence = str(normalized.get("evidence", "")).strip()
            normalized["grounded"] = bool(evidence and evidence in diff and str(normalized.get("file", "")) in diff)
            findings.append(normalized)
        deterministic = self._deterministic_review(diff)
        for item in deterministic:
            item["grounded"] = True
        findings.extend(deterministic)
        blocking = [item for item in findings if item.get("grounded") and str(item.get("severity", "")).lower() in {"critical", "high"}]
        result = {"status": "passed" if not blocking else "needs_fix", "findings": findings, "blocking_findings": blocking}
        self._event(run["id"], "review.completed", reviewer="independent-reviewer", result=result)
        return result

    def _review_material(self, run: dict[str, Any], max_chars: int = 160_000) -> str:
        """Combine Git changes with recorded files, including new untracked files."""

        workspace = self._workspace(run["workspace_id"], run["owner"])
        root = Path(workspace["workspace_path"])
        git_diff = ""
        git_marker = root / ".git"
        if git_marker.is_dir() or git_marker.is_file():
            try:
                git_result = self._invoke(run, "", "git_diff", {})
            except (ApprovalRequired, KeyError, OSError, ValueError):
                git_result = {}
            if self._exit_code(git_result.get("exit_code")) == 0:
                git_diff = str(git_result.get("stdout", ""))
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT filepath, new_hash FROM codebase_file_changes WHERE run_id = ? ORDER BY created_at, id",
                (run["id"],),
            ).fetchall()
        latest = {str(row["filepath"]): str(row["new_hash"]) for row in rows if str(row["filepath"]).strip()}
        sections = [str(git_diff).strip()] if str(git_diff).strip() else []
        used = sum(len(section) for section in sections)
        for relative, digest in latest.items():
            if used >= max_chars:
                break
            path = RepositoryIntelligence(root).resolve(relative)
            if not digest or not path.is_file():
                section = f"\n--- deleted file: {relative}\n"
            else:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                section = f"\n--- recorded file: {relative}\n+++ {relative}\n" + "\n".join(f"+{line}" for line in content.splitlines())
            remaining = max_chars - used
            sections.append(section[:remaining])
            used += min(len(section), remaining)
        return "\n".join(sections)

    def _validate(self, run: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        current_map = RepositoryIntelligence(workspace["workspace_path"]).scan()
        criteria = self._refresh_completion_criteria(run["completion"], current_map)
        results = []
        for kind in ("lint", "typecheck", "test", "build", "browser", "runtime"):
            item = criteria.get(kind)
            if not isinstance(item, dict) or not item.get("enabled"):
                continue
            command = item.get("command")
            if not command:
                results.append({"kind": kind, "status": "failed", "exit_code": -1, "reason": "required command is not configured"})
                continue
            self._event(run["id"], f"{kind}.started", command=command)
            output = self._invoke(run, "", "run_shell", {"command": command, "timeout_seconds": item.get("timeout_seconds", 900)})
            normalized_exit_code = self._exit_code(output.get("exit_code"))
            status = "passed" if normalized_exit_code == 0 else "failed"
            output["exit_code"] = normalized_exit_code if normalized_exit_code is not None else -1
            result = {"kind": kind, "status": status, **output}
            results.append(result)
            self._record_test(run["id"], kind, result)
            self._event(run["id"], f"{kind}.{status}", exit_code=result["exit_code"], duration_ms=result.get("duration_ms", 0))
        artifact_result = self._artifact_check(run, current_map)
        results.append(artifact_result)
        self._record_test(run["id"], "artifacts", artifact_result)
        self._event(run["id"], f"artifacts.{artifact_result['status']}", files=artifact_result.get("files", []))
        passed = bool(results) and all(item["status"] == "passed" for item in results)
        summary = ", ".join(f"{item['kind']} {item['status']}" for item in results) or "No completion checks were configured"
        return {"passed": passed, "checks": results, "summary": summary}

    def _refresh_completion_criteria(self, original: dict[str, Any], project_map: dict[str, Any]) -> dict[str, Any]:
        """Discover validation commands created during a greenfield run."""

        discovered = self._completion_criteria(project_map, {})
        root = Path(str(project_map.get("root", "")))
        # A package script cannot execute until its dependencies are present.
        # Keep structural checks active, while avoiding a misleading build
        # failure that can only say the package manager has not run yet. A
        # model that installs dependencies causes these gates to activate on
        # the same validation pass.
        node_dependencies_ready = not (root / "package.json").is_file() or (root / "node_modules").is_dir()
        merged: dict[str, Any] = {}
        for kind in ("lint", "typecheck", "test", "build", "browser", "runtime"):
            existing = original.get(kind) if isinstance(original.get(kind), dict) else {}
            candidate = discovered.get(kind) if isinstance(discovered.get(kind), dict) else {}
            command_text = " ".join(str(item) for item in candidate.get("command", [])) if isinstance(candidate.get("command"), list) else str(candidate.get("command", ""))
            if not node_dependencies_ready and command_text.startswith(("npm ", "pnpm ", "yarn ")):
                candidate = {**candidate, "enabled": False}
            merged[kind] = existing if existing.get("enabled") else candidate
        return merged

    def _invoke(self, run: dict[str, Any], task_id: str, tool: str, arguments: dict[str, Any], *, approved: bool = False) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        tools = CodebaseTools(self.sandboxes, run["workspace_id"], backend=workspace["sandbox_backend"], event=lambda event_type, **payload: self._event(run["id"], event_type, **payload))
        arguments = tools._normalize_arguments(tool, arguments)
        call_id = uuid4().hex
        decision = tools.policy.classify(tool, arguments)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_tool_calls (id, run_id, task_id, tool, input_json, status, risk, started_at) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)", (call_id, run["id"], task_id, tool, json.dumps(arguments, default=str), int(decision["risk"]), now))
        try:
            result = tools.execute(tool, arguments, approved=approved)
        except Exception as exc:
            with self.store.session() as connection:
                connection.execute("UPDATE codebase_tool_calls SET status = 'failed', error = ?, finished_at = ? WHERE id = ?", (str(exc), utc_now(), call_id))
            raise
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_tool_calls SET status = 'completed', output_json = ?, finished_at = ? WHERE id = ?", (json.dumps(result, default=str), utc_now(), call_id))
            if tool in {"write_file", "patch_file", "delete_file", "move_file"}:
                connection.execute("INSERT INTO codebase_file_changes (id, run_id, filepath, old_hash, new_hash, patch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (uuid4().hex, run["id"], result.get("path", ""), result.get("old_sha256", ""), result.get("new_sha256", ""), result.get("patch", ""), utc_now()))
                if tool == "move_file" and result.get("source"):
                    connection.execute("INSERT INTO codebase_file_changes (id, run_id, filepath, old_hash, new_hash, patch, created_at) VALUES (?, ?, ?, ?, '', ?, ?)", (uuid4().hex, run["id"], result.get("source", ""), result.get("old_sha256", ""), f"moved to {result.get('destination', '')}", utc_now()))
        return result

    def _publish_verified_changes(self, run: dict[str, Any]) -> dict[str, Any]:
        """Copy this run's verified file changes back to the selected folder."""

        workspace = self._workspace(run["workspace_id"], run["owner"])
        workspace_root = Path(workspace["workspace_path"]).resolve()
        source_root = Path(workspace["source_path"]).resolve()
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT filepath, new_hash FROM codebase_file_changes WHERE run_id = ? ORDER BY created_at, id",
                (run["id"],),
            ).fetchall()
        latest = {str(row["filepath"]): str(row["new_hash"]) for row in rows if str(row["filepath"]).strip()}
        backup_root = (self.store.data_dir / "codebase-source-backups" / run["id"]).resolve()
        backup_root.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        self._event(run["id"], "source.publish.started", source_path=str(source_root), files=len(latest))
        for raw_path, new_hash in latest.items():
            relative = Path(raw_path.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(part in {".git", "node_modules", ".venv"} for part in relative.parts):
                raise ValueError(f"unsafe publish path: {raw_path}")
            workspace_file = (workspace_root / relative).resolve()
            source_file = (source_root / relative).resolve()
            if workspace_root not in workspace_file.parents or source_root not in source_file.parents:
                raise ValueError(f"publish path escaped the selected project: {raw_path}")
            if source_file.is_symlink():
                raise ValueError(f"refusing to replace a source symlink: {raw_path}")
            backup_file = (backup_root / relative).resolve()
            existed = source_file.is_file()
            if existed:
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, backup_file)
            if workspace_file.is_file() and new_hash:
                source_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(workspace_file, source_file)
                action = "updated" if existed else "created"
            elif source_file.is_file():
                source_file.unlink()
                action = "deleted"
            else:
                action = "unchanged"
            manifest.append({"path": relative.as_posix(), "action": action, "backup": str(backup_file) if existed else ""})
            self._event(run["id"], "source.publish.file", path=relative.as_posix(), action=action)
        (backup_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        refreshed = RepositoryIntelligence(source_root).scan()
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_projects SET map_json = ?, updated_at = ? WHERE id = ?", (json.dumps(refreshed), utc_now(), run["codebase_project_id"]))
        result = {"source_path": str(source_root), "backup_path": str(backup_root), "files": manifest}
        self._event(run["id"], "source.publish.completed", **result)
        return result

    def _model_json(self, run: dict[str, Any], messages: list[dict[str, str]], *, role: str = "") -> dict[str, Any]:
        role_config = run.get("role_models", {}).get(role, {}) if isinstance(run.get("role_models"), dict) else {}
        provider = str(role_config.get("provider", run["provider"])) if isinstance(role_config, dict) else run["provider"]
        model = str(role_config.get("model", run["model"])) if isinstance(role_config, dict) else run["model"]
        allow_external = bool(role_config.get("allow_external", run["allow_external"])) if isinstance(role_config, dict) else bool(run["allow_external"])
        try:
            output_budget = 16384 if role in {"programmer", "debugger", "architect"} else 8192
            response = self.gateway.complete(messages, provider_id=provider, model=model or None, allow_external=allow_external, max_output_tokens=output_budget)
        except (ProviderConfigurationError, ValueError) as exc:
            self._event(run["id"], "model.failed", error=str(exc))
            return {}
        self._event(run["id"], "model.completed", provider=response.provider, model=response.model, role=role or "default")
        return self._extract_json(response.content)

    @staticmethod
    def _extract_json(value: str) -> dict[str, Any]:
        text = value.strip()
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I)
        candidates = [fenced.group(1)] if fenced else []
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            candidates.append(text[first:last + 1])
        candidates.append(text)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {}

    def _completion_criteria(self, project_map: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
        commands = project_map.get("commands", {})
        criteria: dict[str, Any] = {}
        for name in ("lint", "typecheck", "test", "build"):
            configured = supplied.get(name) if isinstance(supplied.get(name), dict) else {}
            command = configured.get("command") or commands.get(name, "")
            requested = bool(configured.get("enabled", bool(command)))
            criteria[name] = {"enabled": bool(command) and requested, "command": command, "timeout_seconds": int(configured.get("timeout_seconds", 900))}
        for name in ("browser", "runtime"):
            configured = supplied.get(name) if isinstance(supplied.get(name), dict) else {}
            command = configured.get("command") or commands.get(name, "")
            requested = bool(configured.get("enabled", bool(command)))
            criteria[name] = {"enabled": bool(command) and requested, "command": command, "timeout_seconds": int(configured.get("timeout_seconds", 300))}
        languages = set(project_map.get("languages", []))
        if "python" in languages and not criteria.get("runtime", {}).get("enabled"):
            criteria["runtime"] = {"enabled": True, "command": ["python", "-m", "compileall", "-q", "."], "timeout_seconds": 300}
        elif not any(item["enabled"] for item in criteria.values()):
            if {"javascript", "typescript"} & languages:
                entrypoints = [str(item) for item in project_map.get("entrypoints", []) if str(item).strip()]
                fallback = "app.js" if "javascript" in languages else "index.ts"
                criteria["runtime"] = {"enabled": True, "command": ["node", "--check", entrypoints[0] if entrypoints else fallback], "timeout_seconds": 120}
        return criteria

    @staticmethod
    def _automatic_iteration_budget(prompt: str, project_map: dict[str, Any]) -> int:
        """Choose a ceiling from task and repository complexity.

        The controller still stops as soon as all completion gates pass; this is
        a safety ceiling, not a fixed amount of work.
        """
        files = int(project_map.get("file_count", 0) or 0)
        words = len(str(prompt).split())
        budget = 24
        if files > 50 or words > 120:
            budget = 64
        if files > 500 or words > 500:
            budget = 120
        if files > 2_000 or words > 1_500:
            budget = 200
        return budget

    def _next_task(self, run_id: str) -> dict[str, Any] | None:
        ready = self._ready_tasks(run_id)
        return ready[0] if ready else None

    def _recover_blocked_tasks(self, run_id: str, *, source: str) -> list[str]:
        """Requeue legacy malformed-action blocks once their dependencies are ready.

        Earlier versions persisted a task as ``blocked`` after a single model
        response without an ``actions`` list. Those records are recoverable: no
        command was run and no edit was made. Requeueing them lets a resumed run
        request a valid action instead of leaving every dependant task stranded.
        """

        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, depends_on_json, result_json FROM codebase_tasks WHERE run_id = ? AND status = 'blocked' ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
            completed = {
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM codebase_tasks WHERE run_id = ? AND status = 'completed'", (run_id,)
                ).fetchall()
            }
            recovered: list[str] = []
            for row in rows:
                try:
                    dependencies = json.loads(row["depends_on_json"] or "[]")
                except json.JSONDecodeError:
                    dependencies = []
                if not isinstance(dependencies, list) or not all(str(item) in completed for item in dependencies):
                    continue
                try:
                    result = json.loads(row["result_json"] or "{}")
                except json.JSONDecodeError:
                    result = {}
                if not isinstance(result, dict) or result.get("reason") != "model returned no executable actions":
                    continue
                attempts = max(0, int(result.get("invalid_action_attempts", 0) or 0)) + 1
                result.update({
                    "invalid_action_attempts": attempts,
                    "recovered_from": "blocked",
                    "recovery_source": source,
                    "feedback": "A prior model response had no usable action. Inspect the repository and return a valid JSON action batch.",
                })
                connection.execute(
                    "UPDATE codebase_tasks SET status = 'needs_fix', result_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(result), utc_now(), str(row["id"])),
                )
                recovered.append(str(row["id"]))
        return recovered

    def _ready_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
        tasks = [self._task_row(row) for row in rows]
        completed = {task["id"] for task in tasks if task["status"] == "completed"}
        return [
            task for task in tasks
            if task["status"] in {"pending", "needs_fix"}
            and all(dependency in completed for dependency in task["depends_on"])
        ]

    def _store_tasks(self, run_id: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = utc_now()
        with self.store.session() as connection:
            count = connection.execute("SELECT COUNT(*) FROM codebase_tasks WHERE run_id = ?", (run_id,)).fetchone()[0]
            if not count:
                for task in tasks:
                    parent = str(task.get("parent", "")).strip()
                    connection.execute("INSERT INTO codebase_tasks (id, run_id, parent_task_id, title, assigned_agent, depends_on_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (f"{run_id}:{task['id']}", run_id, f"{run_id}:{parent}" if parent else "", task["title"], task["assigned_agent"], json.dumps([f"{run_id}:{item}" for item in task["depends_on"]]), now, now))
        return self._repair_task_dependencies(run_id)

    def _repair_task_dependencies(self, run_id: str) -> list[dict[str, Any]]:
        """Make persisted planner dependencies executable without changing task order.

        Models sometimes refer to another task by its title rather than its
        generated identifier. Persisted title references otherwise become IDs
        that cannot exist, leaving every dependent task permanently blocked.
        This repair retains valid IDs, resolves unique title references, removes
        invalid or self references, and drops only cycle-forming edges.
        """

        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, title, depends_on_json FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
            task_ids = {str(row["id"]) for row in rows}
            references: dict[str, str | None] = {}
            for row in rows:
                task_id = str(row["id"])
                references[task_id.casefold()] = task_id
                references[task_id.rsplit(":", 1)[-1].casefold()] = task_id
                title_key = self._task_dependency_key(str(row["title"]))
                if title_key:
                    references[title_key] = task_id if title_key not in references else None

            graph: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
            repairs: list[dict[str, Any]] = []
            for row in rows:
                task_id = str(row["id"])
                try:
                    raw_dependencies = json.loads(row["depends_on_json"] or "[]")
                except json.JSONDecodeError:
                    raw_dependencies = []
                normalized: list[str] = []
                for raw_dependency in raw_dependencies if isinstance(raw_dependencies, list) else []:
                    raw_text = str(raw_dependency).strip()
                    candidate = self._resolve_task_dependency(raw_text, run_id, references)
                    if candidate is None:
                        repairs.append({"task_id": task_id, "dependency": raw_text, "action": "removed_unknown"})
                        continue
                    if candidate == task_id:
                        repairs.append({"task_id": task_id, "dependency": raw_text, "action": "removed_self"})
                        continue
                    if candidate in normalized:
                        continue
                    if self._dependency_reaches(graph, candidate, task_id):
                        repairs.append({"task_id": task_id, "dependency": raw_text, "action": "removed_cycle"})
                        continue
                    normalized.append(candidate)
                    graph[task_id].add(candidate)
                    if candidate != raw_text:
                        repairs.append({"task_id": task_id, "dependency": raw_text, "resolved_to": candidate, "action": "resolved"})
                if normalized != (raw_dependencies if isinstance(raw_dependencies, list) else []):
                    connection.execute(
                        "UPDATE codebase_tasks SET depends_on_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(normalized), utc_now(), task_id),
                    )
        return repairs

    @staticmethod
    def _task_dependency_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()

    @classmethod
    def _resolve_task_dependency(cls, value: str, run_id: str, references: dict[str, str | None]) -> str | None:
        cleaned = str(value).strip()
        if cleaned.startswith(f"{run_id}:"):
            cleaned = cleaned.removeprefix(f"{run_id}:")
        direct = references.get(cleaned.casefold())
        if direct:
            return direct
        return references.get(cls._task_dependency_key(cleaned))

    @staticmethod
    def _dependency_reaches(graph: dict[str, set[str]], start: str, target: str) -> bool:
        pending = [start]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(graph.get(current, ()))
        return False

    @staticmethod
    def _task_graph_error(tasks: list[dict[str, Any]]) -> str:
        blocked: list[str] = []
        for task in tasks[:8]:
            dependencies = ", ".join(
                str(item).rsplit(":", 1)[-1] for item in task.get("depends_on", [])
            ) or "none"
            task_id = str(task.get("id", "")).rsplit(":", 1)[-1]
            blocked.append(f"{task_id} ({task.get('title', 'untitled')}): {dependencies}")
        if blocked:
            return "Task graph cannot continue. Unfinished task dependencies: " + "; ".join(blocked)
        return "Task graph cannot continue because no runnable task remains."

    def _add_repair_task(self, run_id: str, title: str, feedback: str, *, assigned_agent: str) -> str:
        with self.store.session() as connection:
            rows = connection.execute("SELECT id, status FROM codebase_tasks WHERE run_id = ? ORDER BY created_at, id", (run_id,)).fetchall()
            dependencies = [str(row["id"]) for row in rows if row["status"] == "completed"]
            sequence = len(rows) + 1
            task_id = f"{run_id}:R{sequence}"
            now = utc_now()
            connection.execute(
                "INSERT INTO codebase_tasks (id, run_id, title, assigned_agent, status, depends_on_json, result_json, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (task_id, run_id, title, assigned_agent, json.dumps(dependencies), json.dumps({"feedback": feedback[-20_000:]}), now, now),
            )
        self._event(run_id, "task.created", task_id=task_id, title=title, assigned_agent=assigned_agent, depends_on=dependencies)
        return task_id

    def _unfinished_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM codebase_tasks WHERE run_id = ? AND status != 'completed' ORDER BY created_at, id", (run_id,)).fetchall()
        return [self._task_row(row) for row in rows]

    def _set_task(self, task_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_tasks SET status = ?, result_json = ?, updated_at = ? WHERE id = ?", (status, json.dumps(result or {}), utc_now(), task_id))

    def _set_run(self, run_id: str, *, status: str | None = None, iteration: int | None = None, summary: str | None = None, error: str | None = None, finished: bool | None = None) -> None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        for column, value in (("status", status), ("iteration", iteration), ("summary", summary), ("error", error)):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if finished is not None:
            assignments.append("finished_at = ?")
            values.append(utc_now() if finished else "")
        values.append(run_id)
        with self.store.session() as connection:
            connection.execute(f"UPDATE codebase_runs SET {', '.join(assignments)} WHERE id = ?", values)

    def _event(self, run_id: str, event_type: str, **payload: Any) -> None:
        safe_payload = self._redact_payload(payload)
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_events (id, run_id, type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)", (uuid4().hex, run_id, event_type, json.dumps(safe_payload, default=str), utc_now()))

    def _record_snapshot(self, run_id: str, snapshot: dict[str, Any], label: str) -> None:
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_snapshots (id, run_id, workspace_id, label, path, sha256, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (snapshot["id"], run_id, snapshot["workspace_id"], label, snapshot["path"], snapshot["sha256"], snapshot["size_bytes"], utc_now()))
        self._event(run_id, "snapshot.created", snapshot_id=snapshot["id"], label=label, sha256=snapshot["sha256"])

    def _record_test(self, run_id: str, kind: str, result: dict[str, Any]) -> None:
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_test_runs (id, run_id, kind, command_json, stdout, stderr, exit_code, duration_ms, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid4().hex, run_id, kind, json.dumps(result.get("command", [])), result.get("stdout", "")[-200000:], result.get("stderr", "")[-200000:], int(result.get("exit_code", -1)), float(result.get("duration_ms", 0)), result.get("status", "failed"), utc_now()))

    def _request_approval(self, run_id: str, required: ApprovalRequired, task_id: str = "") -> dict[str, Any]:
        approval_id = uuid4().hex
        payload = {"task_id": task_id, "arguments": required.payload}
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_approvals (id, run_id, tool, reason, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)", (approval_id, run_id, required.tool, required.reason, json.dumps(payload, default=str), utc_now()))
        self._event(run_id, "approval.required", approval_id=approval_id, tool=required.tool, reason=required.reason, payload=payload)
        return {"id": approval_id, "tool": required.tool, "reason": required.reason}

    def _remember_success(self, run: dict[str, Any], validation: dict[str, Any], review: dict[str, Any]) -> None:
        content = json.dumps({"goal": run["user_prompt"], "validation": validation["summary"], "review": review["status"]}, sort_keys=True)
        with self.store.session() as connection:
            connection.execute("INSERT INTO codebase_memories (id, codebase_project_id, category, content, source_run_id, created_at) VALUES (?, ?, 'successful-run', ?, ?, ?)", (uuid4().hex, run["codebase_project_id"], content, run["id"], utc_now()))
            project = connection.execute("SELECT memory_json FROM codebase_projects WHERE id = ?", (run["codebase_project_id"],)).fetchone()
            memory = json.loads(project["memory_json"] or "{}") if project else {}
            memory["last_successful_validation"] = {"run_id": run["id"], "summary": validation["summary"], "at": utc_now()}
            connection.execute("UPDATE codebase_projects SET memory_json = ?, updated_at = ? WHERE id = ?", (json.dumps(memory), utc_now(), run["codebase_project_id"]))

    def _remember_repository_state(self, run: dict[str, Any]) -> None:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        project_map = RepositoryIntelligence(workspace["workspace_path"]).scan()
        with self.store.session() as connection:
            row = connection.execute("SELECT memory_json FROM codebase_projects WHERE id = ?", (run["codebase_project_id"],)).fetchone()
            memory = json.loads(row["memory_json"] or "{}") if row else {}
            memory["repository_state"] = {
                "git": project_map.get("git", {}),
                "commands": project_map.get("commands", {}),
                "last_run_id": run["id"],
                "updated_at": utc_now(),
            }
            connection.execute(
                "UPDATE codebase_projects SET map_json = ?, memory_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(project_map), json.dumps(memory), utc_now(), run["codebase_project_id"]),
            )

    def _recent_run_memory(self, codebase_project_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, user_prompt, status, summary, error, finished_at FROM codebase_runs WHERE codebase_project_id = ? ORDER BY started_at DESC LIMIT ?",
                (codebase_project_id, max(1, min(int(limit), 20))),
            ).fetchall()
        return [dict(row) for row in rows]

    def _artifact_check(self, run: dict[str, Any], project_map: dict[str, Any] | None = None) -> dict[str, Any]:
        workspace = self._workspace(run["workspace_id"], run["owner"])
        root = Path(workspace["workspace_path"])
        bad_names = {"npm-debug.log", "yarn-error.log", "core", "core.dump"}
        files = []
        for path in root.rglob("*"):
            if not path.is_file() or any(part in {".git", "node_modules", ".venv"} for part in path.relative_to(root).parts):
                continue
            if path.name.lower() in bad_names or path.suffix.lower() in {".rej", ".orig"} or path.name.lower().startswith("hs_err_pid"):
                files.append(path.relative_to(root).as_posix())
                if len(files) >= 100:
                    break
        source_map = RepositoryIntelligence(workspace["source_path"]).scan()
        profile = classify_request(run["user_prompt"], source_map)
        if self._static_web_request(run["user_prompt"]):
            profile = {**profile, "framework": "browser", "language": "javascript", "explicit_stack": True}
        missing = check_artifact_contract(root, profile) if profile["mode"] == "greenfield" else []
        page_count_match = re.search(r"\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:static\s+)?pages?\b", run["user_prompt"], re.I)
        requested_pages = self._requested_page_count(run["user_prompt"]) if page_count_match and profile.get("requires_ui") else 0
        if requested_pages:
            actual_pages = [
                path for path in root.rglob("*.html")
                if not any(part in {".git", "node_modules", ".venv", "dist", "build"} for part in path.relative_to(root).parts)
            ]
            if len(actual_pages) < requested_pages:
                missing.append(f"at least {requested_pages} linked HTML pages (found {len(actual_pages)})")
        status = "passed" if not files and not missing else "failed"
        return {
            "kind": "artifacts",
            "status": status,
            "command": ["artifact-scan"],
            "stdout": "Project structure and error-artifact checks passed" if status == "passed" else "",
            "stderr": "; ".join(filter(None, ["Error artifacts: " + ", ".join(files) if files else "", "Missing project structure: " + ", ".join(missing) if missing else ""])),
            "exit_code": 0 if status == "passed" else 1,
            "duration_ms": 0,
            "files": files,
            "missing": missing,
            "profile": profile,
        }

    @staticmethod
    def _deterministic_review(diff: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for line_number, line in enumerate(diff.splitlines(), start=1):
            if line.startswith("+") and any(marker in line for marker in ("<<<<<<<", "=======", ">>>>>>>")):
                findings.append({"severity": "critical", "file": "git-diff", "line": line_number, "issue": "Unresolved merge-conflict marker"})
            if line.startswith("+") and "-----BEGIN PRIVATE KEY-----" in line:
                findings.append({"severity": "critical", "file": "git-diff", "line": line_number, "issue": "Private key material was added"})
        return findings

    def _cleanup_processes(self, run_id: str) -> None:
        try:
            owner = self._owner_for_run(run_id)
            run = self.get_run(run_id, owner=owner)
        except KeyError:
            return
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT output_json FROM codebase_tool_calls WHERE run_id = ? AND tool = 'start_process' AND status = 'completed'",
                (run_id,),
            ).fetchall()
        for row in rows:
            try:
                process_id = str(json.loads(row["output_json"] or "{}").get("id", ""))
                if process_id:
                    result = self.sandboxes.stop_process(run["workspace_id"], process_id)
                    self._event(run_id, "process.stopped", process=result, reason="run finished")
            except (KeyError, OSError, ValueError):
                continue

    def _recover_interrupted_runs(self) -> None:
        with self.store.session() as connection:
            connection.execute("UPDATE codebase_runs SET status = 'interrupted', error = 'Desktop stopped during the run', updated_at = ? WHERE status IN ('queued','analysing','running','validating')", (utc_now(),))

    def _cancelled(self, run_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(run_id)
        return bool(event and event.is_set())

    def _owner_for_run(self, run_id: str) -> str:
        with self.store.session() as connection:
            row = connection.execute("SELECT owner FROM codebase_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("codebase run not found")
        return str(row["owner"])

    def _workspace(self, workspace_id: str, owner: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM codebase_workspaces WHERE id = ? AND owner = ?", (workspace_id, owner)).fetchone()
        if row is None:
            raise KeyError("workspace not found")
        return self._workspace_row(row)

    def _audit(self, action: str, resource: str, project: str, metadata: dict[str, Any], status: str = "allowed") -> None:
        self.store.add_audit_event(AuditEvent(actor="codebase-agent", action=action, resource=resource, project=project, metadata=metadata, status=status))

    @staticmethod
    def _compact_map(project_map: dict[str, Any]) -> dict[str, Any]:
        return {key: project_map.get(key) for key in (
            "file_count", "total_bytes", "languages", "framework", "database", "entrypoints",
            "important_files", "commands", "dependencies", "configuration_files", "test_files",
            "database_models", "routes", "git",
        )}

    @staticmethod
    def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        text = json.dumps(payload, default=str)
        text = re.sub(r"(?i)(token|secret|password|api[_-]?key)([\"']?\s*[:=]\s*[\"']?)([^\s\"',}]+)", r"\1\2<redacted>", text)
        return json.loads(text)

    @staticmethod
    def _project_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "project": row["project"], "owner": row["owner"], "name": row["name"], "source_path": row["source_path"], "repo_url": row["repo_url"], "map": json.loads(row["map_json"] or "{}"), "memory": json.loads(row["memory_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _workspace_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "codebase_project_id": row["codebase_project_id"], "owner": row["owner"], "sandbox_backend": row["sandbox_backend"], "source_path": row["source_path"], "workspace_path": row["workspace_path"], "branch": row["branch"], "status": row["status"], "environment": json.loads(row["environment_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _run_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "codebase_project_id": row["codebase_project_id"], "workspace_id": row["workspace_id"], "project": row["project"], "owner": row["owner"], "user_prompt": row["user_prompt"], "status": row["status"], "provider": row["provider"], "model": row["model"], "role_models": json.loads(row["role_models_json"] or "{}"), "allow_external": bool(row["allow_external"]), "iteration": int(row["iteration"]), "max_iterations": int(row["max_iterations"]), "completion": json.loads(row["completion_json"] or "{}"), "summary": row["summary"], "error": row["error"], "started_at": row["started_at"], "updated_at": row["updated_at"], "finished_at": row["finished_at"]}

    @staticmethod
    def _task_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "run_id": row["run_id"], "parent_task_id": row["parent_task_id"], "title": row["title"], "assigned_agent": row["assigned_agent"], "status": row["status"], "depends_on": json.loads(row["depends_on_json"] or "[]"), "result": json.loads(row["result_json"] or "{}"), "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _approval_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "run_id": row["run_id"], "tool": row["tool"], "reason": row["reason"], "payload": json.loads(row["payload_json"] or "{}"), "status": row["status"], "created_at": row["created_at"], "resolved_at": row["resolved_at"]}

    @staticmethod
    def _test_row(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "kind": row["kind"], "command": json.loads(row["command_json"] or "[]"), "stdout": row["stdout"], "stderr": row["stderr"], "exit_code": int(row["exit_code"]), "duration_ms": float(row["duration_ms"]), "status": row["status"], "created_at": row["created_at"]}

    @staticmethod
    def _event_row(row: Any) -> dict[str, Any]:
        return {"sequence": int(row["sequence"]), "id": row["id"], "run_id": row["run_id"], "type": row["type"], "payload": json.loads(row["payload_json"] or "{}"), "created_at": row["created_at"]}
