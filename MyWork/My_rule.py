# Excel を読み、Web から価格データを取得して更新する

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime

import requests
import schedule
from openpyxl import load_workbook

# --- ログ ---
def setup_logging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"price_update_log_{datetime.now():%Y%m}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

WORKBOOK_FILENAME = "maki_250630.xlsx"


def resolve_workbook_path():
    """リポジトリ直下実行・MyWork 配下実行の両方で探す。"""
    candidates = [
        os.path.join("MyWork", "datasets", WORKBOOK_FILENAME),
        os.path.join("datasets", WORKBOOK_FILENAME),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("."):
        if WORKBOOK_FILENAME in files:
            return os.path.join(root, WORKBOOK_FILENAME)
    return None


def get_usd_jpy_rate():
    """Frankfurter API から USD/JPY を取得。失敗時はデフォルト。"""
    url = "https://api.frankfurter.app/latest?from=USD&to=JPY"
    default = 158.0
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "rates" not in data or "JPY" not in data["rates"]:
            raise ValueError("rates.JPY がありません")
        rate = float(data["rates"]["JPY"])
        if rate <= 0:
            raise ValueError(f"レートが不正です: {rate}")
        date = data.get("date", "")
        logger.info("USD/JPY = %.4f (Frankfurter, date=%s)", rate, date)
        return rate
    except Exception as e:
        logger.warning("Frankfurter API 取得エラー: %s。デフォルト %.1f を使用します。", e, default)
        return default


def get_crypto_prices():
    """CoinGecko から BTC / ETH（万円）。"""
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bitcoin,ethereum&vs_currencies=jpy"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "bitcoin" not in data or "ethereum" not in data:
            raise ValueError("必要なキーがありません")
        btc = data["bitcoin"]["jpy"] / 10000
        eth = data["ethereum"]["jpy"] / 10000
        if btc <= 0 or eth <= 0:
            raise ValueError("価格が不正です")
        return {"BTC": btc, "ETH": eth}
    except Exception as e:
        logger.warning("CoinGecko 取得エラー: %s。デフォルト価格を使用します。", e)
        return {"BTC": 600.0, "ETH": 40.0}


def get_precious_metal_prices():
    """MetalPriceAPI から金・銀・プラチナ（円/g）。失敗時はデフォルト。"""
    defaults = {"gold": 22000.0, "platinum": 8000.0, "silver": 286.0}
    api_key = "0f87f8e218bb8fb3c0b40c46d56911a8"
    api_url = "https://api.metalpriceapi.com/v1/latest"
    oz_to_gram = 28.35

    try:
        usd_jpy = get_usd_jpy_rate()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "x-api-key": api_key,
        }
        r = requests.get(
            api_url,
            headers=headers,
            params={"base": "USD", "currencies": "XAU,XAG,XPT"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if "rates" not in data:
            raise ValueError("rates がありません")

        rates = data["rates"]
        if isinstance(rates, str):
            rates = json.loads(rates)

        prices = {}
        if "USDXAU" in rates:
            prices["gold"] = float(rates["USDXAU"]) * usd_jpy / oz_to_gram
        if "USDXAG" in rates:
            prices["silver"] = float(rates["USDXAG"]) * usd_jpy / oz_to_gram
        if "USDXPT" in rates:
            prices["platinum"] = float(rates["USDXPT"]) * usd_jpy / oz_to_gram

        if len(prices) < 3:
            raise ValueError("貴金属が揃いませんでした")
        return prices
    except Exception as e:
        logger.warning("貴金属 API 取得エラー: %s。デフォルトを使用します。", e)
        return defaults.copy()


def _apply_metal_price_guard(p_old, p_new, metal_name: str, max_relative_delta: float = 0.1):
    """
    |(Pr - P) / P| が max_relative_delta 未満のときだけ Pr を採用。
    以上のときは P をそのまま維持。P が比較不能（None・非数値・0）のときは Pr。
    """
    pr = float(p_new)
    if p_old is None:
        return pr
    try:
        p = float(p_old)
    except (TypeError, ValueError):
        return pr
    if p == 0:
        return pr
    rel = abs((pr - p) / p)
    if rel >= max_relative_delta:
        logger.warning(
            "%s: 相対変化 %.2f%% (閾値 %.0f%%) のため現値 %.2f を維持（取得 %.2f）",
            metal_name,
            rel * 100,
            max_relative_delta * 100,
            p,
            pr,
        )
        return p
    return pr


def update_crypto_prices_in_excel(file_path):
    crypto = get_crypto_prices()
    wb = None
    try:
        wb = load_workbook(file_path)
        if "calc" not in wb.sheetnames:
            logger.error("'calc' シートがありません")
            return
        ws = wb["calc"]
        ws["J29"].value = crypto["BTC"]
        ws["J30"].value = crypto["ETH"]
        logger.info("仮想通貨: J29=BTC %.2f 万円, J30=ETH %.2f 万円", crypto["BTC"], crypto["ETH"])
        wb.save(file_path)
    finally:
        if wb is not None:
            wb.close()


def update_precious_metal_prices_in_excel(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    metal = get_precious_metal_prices()
    if not all(k in metal for k in ("gold", "silver", "platinum")):
        raise ValueError("貴金属データが不完全です")

    wb = None
    try:
        wb = load_workbook(file_path)
        if "ストック" not in wb.sheetnames:
            raise ValueError(f"'ストック' シートがありません: {wb.sheetnames}")

        ws = wb["ストック"]
        # 金 R31 / 銀 R32 / プラチナ R33（|(Pr-P)/P| >= 0.1 のときは現値 P を維持）
        gold_v = _apply_metal_price_guard(ws["R31"].value, metal["gold"], "金")
        silver_v = _apply_metal_price_guard(ws["R32"].value, metal["silver"], "銀")
        platinum_v = _apply_metal_price_guard(ws["R33"].value, metal["platinum"], "白金")
        ws["R31"].value = gold_v
        ws["R32"].value = silver_v
        ws["R33"].value = platinum_v
        logger.info(
            "貴金属: R31=金 %.2f, R32=銀 %.2f, R33=白金 %.2f 円/g",
            gold_v,
            silver_v,
            platinum_v,
        )
        wb.save(file_path)
    finally:
        if wb is not None:
            wb.close()


def update_execution_date_in_excel(file_path):
    date_str = datetime.now().strftime("%Y/%m/%d")
    wb = None
    try:
        wb = load_workbook(file_path)
        if "家計資産" not in wb.sheetnames:
            logger.error("'家計資産' シートがありません")
            return
        ws = wb["家計資産"]
        ws["H15"].value = date_str
        ws["H17"].value = date_str
        logger.info("実行日: H15/H17 = %s", date_str)
        wb.save(file_path)
    finally:
        if wb is not None:
            wb.close()


def print_workbook_snapshot(file_path):
    """更新後の主要セルを openpyxl のみで表示（pandas 不要）。"""
    wb = None
    try:
        wb = load_workbook(file_path, read_only=True)
        if "calc" in wb.sheetnames:
            c = wb["calc"]
            logger.info("確認 calc: J29=%s J30=%s", c["J29"].value, c["J30"].value)
        if "ストック" in wb.sheetnames:
            s = wb["ストック"]
            logger.info(
                "確認 ストック: R31=%s R32=%s R33=%s",
                s["R31"].value,
                s["R32"].value,
                s["R33"].value,
            )
        if "家計資産" in wb.sheetnames:
            k = wb["家計資産"]
            logger.info("確認 家計資産: H15=%s H17=%s", k["H15"].value, k["H17"].value)
    finally:
        if wb is not None:
            wb.close()


def update_all_prices_with_logging(file_path):
    logger.info("=== 価格データの自動更新を開始 ===")
    if not os.path.exists(file_path):
        logger.error("ファイルが見つかりません: %s", file_path)
        return False

    ok = 0
    tasks = (
        ("仮想通貨", update_crypto_prices_in_excel),
        ("貴金属", update_precious_metal_prices_in_excel),
        ("実行日", update_execution_date_in_excel),
    )
    for name, fn in tasks:
        logger.info("--- %s ---", name)
        try:
            fn(file_path)
            ok += 1
            logger.info("%s 完了", name)
        except Exception as e:
            logger.error("%s エラー: %s", name, e)
            logger.error(traceback.format_exc())

    logger.info("=== 更新完了: %s/%s タスク成功 ===", ok, len(tasks))
    return ok == len(tasks)


def scheduled_job():
    path = resolve_workbook_path() or os.path.join("datasets", WORKBOOK_FILENAME)
    logger.info("定期実行ジョブ開始 path=%s", path)
    try:
        success = update_all_prices_with_logging(path)
        if success:
            logger.info("定期実行ジョブ正常終了")
        else:
            logger.warning("定期実行ジョブは一部失敗")
    except Exception as e:
        logger.error("定期実行ジョブ例外: %s", e)


def run_scheduler():
    logger.info("スケジューラ開始（毎日 3:00）。初回に 1 回実行します。")
    schedule.every().day.at("03:00").do(scheduled_job)
    scheduled_job()
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("スケジューラを停止しました")


def create_batch_file():
    content = f'''@echo off
cd /d "{os.getcwd()}"
python My_rule.py --auto
pause
'''
    path = "run_price_update.bat"
    try:
        with open(path, "w", encoding="shift_jis") as f:
            f.write(content)
        logger.info("バッチを作成: %s", path)
        return path
    except Exception as e:
        logger.error("バッチ作成エラー: %s", e)
        return None


def show_task_scheduler_instructions():
    print(
        "タスク スケジューラ: 基本タスクで毎日時刻を指定し、"
        "プログラムに run_price_update.bat（または python My_rule.py --auto）を登録。"
        "詳細は Microsoft のドキュメントを参照。"
    )


if __name__ == "__main__":
    file_path = resolve_workbook_path()
    if not file_path:
        logger.error("%s が見つかりません", WORKBOOK_FILENAME)
        sys.exit(1)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--scheduler":
            run_scheduler()
        elif arg == "--auto":
            update_all_prices_with_logging(file_path)
        elif arg == "--setup":
            logger.info("セットアップ")
            if create_batch_file():
                show_task_scheduler_instructions()
        else:
            print("使い方: python My_rule.py | --auto | --scheduler | --setup")
    else:
        logger.info("=== 価格更新（手動） ===")
        update_crypto_prices_in_excel(file_path)
        try:
            update_precious_metal_prices_in_excel(file_path)
        except Exception as e:
            logger.error("貴金属更新: %s\n%s", e, traceback.format_exc())
        update_execution_date_in_excel(file_path)
        print_workbook_snapshot(file_path)
