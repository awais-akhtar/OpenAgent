import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Calendar event cards retain canonical CSS geometry (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(selectedTheme => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, theme);
      await page.goto('/?view=calendar');

      const event = page.locator('.calendar-grid .calendar-event').first();
      await expect(event).toBeVisible();
      const style = await event.evaluate(element => {
        const computed = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return {
          padding: computed.padding,
          borderLeftWidth: computed.borderLeftWidth,
          borderLeftStyle: computed.borderLeftStyle,
          borderTopStyle: computed.borderTopStyle,
          borderRadius: computed.borderRadius,
          boxShadow: computed.boxShadow,
          width: rect.width,
          height: rect.height,
          xOffset: rect.x - element.parentElement!.getBoundingClientRect().x,
          yOffset: rect.y - element.parentElement!.getBoundingClientRect().y,
          parentWidth: element.parentElement!.getBoundingClientRect().width,
          borderTopColor: computed.borderTopColor,
        };
      });

      // These values come from the immutable approved HTML's .calendar-event rule.
      expect(style.padding).toBe('7px 8px');
      expect(style.borderLeftWidth).toBe('3px');
      expect(style.borderLeftStyle).toBe('solid');
      expect(style.borderTopStyle).toBe('outset');
      expect(style.borderTopColor).toBe(theme === 'dark' ? 'rgb(255, 255, 255)' : 'rgb(0, 0, 0)');
      expect(style.borderRadius).toBe('6px');
      expect(style.boxShadow).toContain('rgba(22, 58, 101, 0.125)');
      expect(style.width + 6).toBe(style.parentWidth);
      expect(style.height).toBe(62);
      expect(style.xOffset).toBe(3);
      expect(style.yOffset).toBe(2);
      expect(apiRequests).toEqual([]);
    });
  }
}
