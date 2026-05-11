import re
import time
import smtplib
import logging
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright


URL = "https://www.ur-net.go.jp/chintai/kanto/tokyo/list/"

# ===== メール設定 =====
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "jojo1139@gmail.com"
SMTP_PASSWORD = "vdmrishmmzdkazea"
MAIL_FROM = SMTP_USER
MAIL_TO = "maki_jun@hotmail.com"

# ===== 監視設定 =====
# ページ上のエリア名（見出し文言）と、直後に現れる js-area-room の件数を対応付けて監視
AREAS = ("都心", "23区南")

DAY_INTERVAL_SEC = 10 * 60      # 8:00-18:00
NIGHT_INTERVAL_SEC = 2 * 60 * 60  # 18:00-8:00

LOG_FILE = Path(__file__).with_name("UR_difference_check.log")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ur_difference_check")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_interval_seconds() -> int:
    now_hour = datetime.now().hour
    if 8 <= now_hour < 18:
        return DAY_INTERVAL_SEC
    return NIGHT_INTERVAL_SEC


def send_mail(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASSWORD)
        smtp.send_message(msg)


def extract_area_counts(page) -> dict[str, str | None]:
    """各エリア見出しの近傍にある js-area-room の件数を取得する。"""
    html = page.content()
    result: dict[str, str | None] = {}
    for label in AREAS:
        m = re.search(
            re.escape(label) + r"[\s\S]{0,4000}?js-area-room\">(\d+)<",
            html,
        )
        result[label] = m.group(1) if m else None
    return result


def main() -> None:
    previous_values: dict[str, str | None] = {a: None for a in AREAS}
    logger = setup_logger()
    logger.info("監視を起動しました。URL=%s, 監視対象=%s", URL, ", ".join(AREAS))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while True:
            try:
                page.goto(URL, wait_until="load", timeout=60000)
                page.wait_for_timeout(3000)

                current = extract_area_counts(page)

                for area in AREAS:
                    v = current[area]
                    if v is None:
                        logger.warning("%s 取得失敗", area)
                    else:
                        logger.info("%s = %s", area, v)

                changes: list[tuple[str, str, str]] = []
                for area in AREAS:
                    cur = current[area]
                    if cur is None:
                        continue
                    prev = previous_values[area]
                    if prev is not None and cur != prev:
                        changes.append((area, prev, cur))

                if changes:
                    parts = [f"{a}: {p} -> {c}" for a, p, c in changes]
                    subject = "UR件数変化: " + "; ".join(parts)
                    body_lines = [
                        "UR賃貸（東京都一覧）のエリア別件数表示が変化しました。",
                        "",
                        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "",
                    ]
                    for area, prev, cur in changes:
                        body_lines.append(f"{area}: {prev} -> {cur}")
                    body_lines.extend(["", f"URL: {URL}"])
                    send_mail(subject, "\n".join(body_lines))
                    logger.info("メール送信完了 (%s)", subject)

                for area in AREAS:
                    cur = current[area]
                    if cur is not None:
                        previous_values[area] = cur

            except Exception as e:
                logger.exception("エラーが発生しました: %s", e)

            sleep_sec = get_interval_seconds()
            logger.info("次回チェックまで %s 分待機", sleep_sec // 60)
            time.sleep(sleep_sec)


if __name__ == "__main__":
    main()