from playwright.sync_api import Page, expect, sync_playwright
import time

def test_deploy_polymer_x(page: Page):
    # 1. Arrange: Go to the app
    page.goto("http://localhost:5173")

    # 2. Act: Click Deploy
    # The button has text "DEPLOY POLYMER-X"
    deploy_btn = page.get_by_role("button", name="DEPLOY POLYMER-X")
    deploy_btn.click()

    # 3. Assert: Wait for deployment result
    # It takes about 1.5s
    # Expect "Deployment Ready" to appear
    expect(page.get_by_text("Deployment Ready")).to_be_visible(timeout=10000)

    # 4. Wait for canvas animation to run a bit so we see the agents
    # The canvas is overlaying everything.
    time.sleep(2)

    # 5. Screenshot
    page.screenshot(path="/home/jules/verification/verification.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Set viewport to something reasonable
        page.set_viewport_size({"width": 1280, "height": 720})
        try:
            test_deploy_polymer_x(page)
            print("Verification script ran successfully.")
        except Exception as e:
            print(f"Verification script failed: {e}")
            page.screenshot(path="/home/jules/verification/failure.png")
        finally:
            browser.close()
