"""Open each Streamlit Community Cloud app in a headless browser so it doesn't go to sleep.

Free Streamlit apps sleep after a period without visitors. A plain HTTP ping doesn't count,
so this loads the page like a real visitor and clicks the wake-up button if the app is asleep.
Run by .github/workflows/keep-awake.yml.

    pip install playwright && playwright install chromium
    python scripts/keep_awake.py https://wnba-agentic-rag.streamlit.app/ ...
"""

import sys

from playwright.sync_api import sync_playwright

WAKE_BUTTON = "button:has-text('get this app back up')"


def visit(page, url: str) -> bool:
    page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(10_000)
    woke = False
    if page.locator(WAKE_BUTTON).count():
        page.locator(WAKE_BUTTON).first.click()
        woke = True
        page.wait_for_timeout(90_000)  # give the app time to boot
        page.reload(wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(10_000)
    still_asleep = page.locator(WAKE_BUTTON).count() > 0
    print(f"{url}: {'woke it up' if woke else 'already awake'}"
          f"{' -- STILL ASLEEP' if still_asleep else ''}", flush=True)
    return not still_asleep


def main(urls: list[str]) -> int:
    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url in urls:
            try:
                ok &= visit(browser.new_page(), url)
            except Exception as e:  # keep going so one bad app doesn't skip the others
                print(f"{url}: error {e}", flush=True)
                ok = False
        browser.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
