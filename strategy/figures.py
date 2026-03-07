import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


# ---------------------------------------------------------
# helper: export folium map to PNG
# ---------------------------------------------------------


def _save_map_png(m, filename, width=900, height=900):

    os.makedirs("img", exist_ok=True)

    tmp_html = "img/tmp_map.html"
    m.save(tmp_html)

    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"--window-size={width},{height}")

    driver = webdriver.Chrome(options=options)

    driver.get("file://" + os.path.abspath(tmp_html))

    time.sleep(2)

    driver.save_screenshot(filename)

    driver.quit()

    os.remove(tmp_html)

    print(f"Saved {filename}")
