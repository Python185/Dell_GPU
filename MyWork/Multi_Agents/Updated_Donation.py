from openai import OpenAI, APIError
import os
from datetime import datetime

client = OpenAI()

# 使用可能モデル（各役割で切り替え）
AVAILABLE_MODELS = ("gpt-4o-mini", "gpt-5.6-sol")
# temperature を指定できないモデル（推論トークンを別途消費する）
REASONING_MODELS = {"gpt-5.6-sol"}

# 各役割のモデル（"gpt-4o-mini" または "gpt-5.6-sol"）
ROLE_MODELS = {
    "Visionary": "gpt-5.6-sol",
    "Analyst": "gpt-4o-mini",
    "Skeptic": "gpt-4o-mini",
    "Legal": "gpt-4o-mini",
    "Donor": "gpt-4o-mini",
    "Director": "gpt-5.6-sol",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------
# 1. Agent prompts
# ---------------------------------------------------------
PROMPTS = {

    "Visionary": """
あなたは革新的な社会システムの構想を考えるVisionaryです。
与えられたテーマについて、既存制度の延長だけに限定せず、
互いに性格の異なるアイデアを3つ提案してください。

各案について以下を簡潔に示してください。
- アイデア
- 誰にどんな価値があるか
- 従来方式との違い
""",

    "Analyst": """
あなたは社会事業・ビジネスモデル分析の専門家です。
提示された各アイデアについて、
成立するための条件を分析してください。

特に以下を評価してください。
- 資金の流れ
- 運営主体
- 利用者が参加するインセンティブ
- 継続可能性
- 小規模な実証実験が可能か
""",

    "Skeptic": """
あなたは非常に批判的なSkepticです。
提示されたアイデアと成立条件について、
失敗する可能性を積極的に探してください。

特に、
- 利用者が集まらない理由
- 運営上の問題
- 悪用・不正
- 既存サービスとの差別化不足
- 見落としている前提
を検討してください。

単なる否定ではなく、致命的問題と改善可能な問題を区別してください。
""",

    "Legal": """
あなたは日本の寄付・NPO・資金移動制度について
法務上の論点を洗い出す専門家です。

これは正式な法律判断ではありません。
最新法令を確認していない事項について断定してはいけません。

各アイデアについて、
- 関係する可能性がある法制度
- 法務上確認すべき事項
- 特に重大と思われるリスク
を簡潔に整理してください。

最後に法務確認優先度を1～10で付けてください。
10は「専門家による確認を最優先すべき」を意味します。
""",

    "Donor": """
あなたは20代の寄付未経験者です。
提示された仕組みを一般ユーザーとして評価してください。

以下について率直に答えてください。
- 使ってみたいか
- 理解しやすいか
- 面倒に感じる部分
- 怪しいと感じる部分
- どう変われば使いたくなるか

専門家的な分析はせず、利用者として反応してください。
""",

    "Director": """
あなたはこのAIシンクタンクのDirectorです。

Visionary、Analyst、Skeptic、Legal、Donorの意見を統合し、
次に検討する価値のある案を決定してください。

重要：
元の案をそのまま選ぶ必要はありません。
複数案の長所を組み合わせたFusion案の方が優れている場合は、
新しい案を提案してください。

以下の観点で評価してください。
- 社会的インパクト
- 新規性
- 実現可能性
- 市民が参加する可能性
- 小規模実証のしやすさ

最後に、
1. 最有望案
2. 選定理由
3. 最大の未解決問題
4. 次に調査・検証すべきこと3項目以内
を示してください。
"""
}


# ---------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------
def _format_role_models(role_models):
    return "\n".join(f"  {role}: {model}" for role, model in role_models.items())


def _completion_budget(model, max_tokens):
    """推論モデルは reasoning トークンも上限に含まれるため余裕を持たせる。"""
    if model in REASONING_MODELS:
        return max(max_tokens * 3, max_tokens + 2000)
    return max_tokens


def call_llm(system_prompt, user_prompt, model,
             max_tokens=600, temperature=0.6):

    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model}. Choose from {AVAILABLE_MODELS}"
        )

    try:
        request_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": _completion_budget(model, max_tokens),
        }
        if model in REASONING_MODELS:
            request_kwargs["reasoning_effort"] = "low"
        else:
            request_kwargs["temperature"] = temperature

        response = client.chat.completions.create(**request_kwargs)
    except APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    choice = response.choices[0]
    content = choice.message.content
    if not content or not content.strip():
        reasoning_tokens = None
        if response.usage and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        raise RuntimeError(
            "OpenAI API returned empty content "
            f"(model={model}, finish_reason={choice.finish_reason}, "
            f"reasoning_tokens={reasoning_tokens})"
        )

    return content


def save_results(theme, ideas, analysis, criticism, legal, donor, final_report,
                 role_models):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(RESULTS_DIR, f"think_tank_{timestamp}.txt")
    director_path = os.path.join(RESULTS_DIR, f"director_{timestamp}.txt")
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models_text = _format_role_models(role_models)

    full_content = f"""=== AI Think Tank ===
日時: {run_at}
モデル:
{models_text}

課題:
{theme}

### Visionary
{ideas}

### Analyst
{analysis}

### Skeptic
{criticism}

### Legal
{legal}

### Donor Persona
{donor}
"""
    director_content = f"""=== Director Summary ===
日時: {run_at}
モデル:
{models_text}

課題:
{theme}

### Director
{final_report}
"""
    log_path = os.path.join(RESULTS_DIR, "課題_log.txt")
    log_entry = f"""=== {run_at} ===
課題:
{theme}

全文: {os.path.basename(full_path)}
Director: {os.path.basename(director_path)}

"""
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    with open(director_path, "w", encoding="utf-8") as f:
        f.write(director_content)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return full_path, director_path


# ---------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------
def run_think_tank_pipeline(theme, role_models=None):

    if role_models is None:
        role_models = ROLE_MODELS.copy()
    else:
        unknown = set(role_models) - set(ROLE_MODELS)
        if unknown:
            raise ValueError(f"Unknown roles: {unknown}")
        role_models = {**ROLE_MODELS, **role_models}

    print(f"\n=== AI Think Tank: {theme} ===\n")
    print("モデル:")
    print(_format_role_models(role_models))
    print()

    # Stage 1: Idea generation
    ideas = call_llm(
        PROMPTS["Visionary"],
        f"テーマ:\n{theme}",
        model=role_models["Visionary"],
        max_tokens=700,
        temperature=0.9
    )

    print("### Visionary")
    print(ideas)


    # Stage 2: Feasibility
    analysis = call_llm(
        PROMPTS["Analyst"],
        f"テーマ:\n{theme}\n\nアイデア:\n{ideas}",
        model=role_models["Analyst"],
        max_tokens=600
    )

    print("\n### Analyst")
    print(analysis)


    # Stage 3: Criticism
    criticism = call_llm(
        PROMPTS["Skeptic"],
        f"アイデア:\n{ideas}\n\n成立条件:\n{analysis}",
        model=role_models["Skeptic"],
        max_tokens=600
    )

    print("\n### Skeptic")
    print(criticism)


    # Stage 4A: Legal screening
    legal = call_llm(
        PROMPTS["Legal"],
        f"テーマ:\n{theme}\n\nアイデア:\n{ideas}",
        model=role_models["Legal"],
        max_tokens=400,
        temperature=0.2
    )

    print("\n### Legal")
    print(legal)


    # Stage 4B: Independent user reaction
    # Skepticの意見は意図的に見せない
    donor = call_llm(
        PROMPTS["Donor"],
        f"以下のアイデアを読んで評価してください。\n\n{ideas}",
        model=role_models["Donor"],
        max_tokens=400,
        temperature=0.7
    )

    print("\n### Donor Persona")
    print(donor)


    # Stage 5: Director decision + Fusion
    full_context = f"""
テーマ:
{theme}

[VISIONARY]
{ideas}

[ANALYST]
{analysis}

[SKEPTIC]
{criticism}

[LEGAL]
{legal}

[DONOR]
{donor}
"""

    final_report = call_llm(
        PROMPTS["Director"],
        full_context,
        model=role_models["Director"],
        max_tokens=800,
        temperature=0.5
    )

    print("\n" + "=" * 60)
    print("### Director")
    print(final_report)

    full_path, director_path = save_results(
        theme, ideas, analysis, criticism, legal, donor, final_report,
        role_models,
    )
    print(f"\n結果を保存しました:")
    print(f"  全文: {full_path}")
    print(f"  Director: {director_path}")
    print(f"  課題ログ: {os.path.join(RESULTS_DIR, '課題_log.txt')}")

    return final_report


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
# 課題（毎回ここを修正）
THEME = (
    "寄付を一部の富裕層や大規模財団だけのものではなく、"
    "一般市民も社会的資源配分に参加できる仕組みにする"
    "#1において、クラウドファンディング型プラットフォームとポイントの融合が出た"
    "#2において、地域プロジェクト型の投資プラットフォームが出た"
    "#3において、企業連携型社会投資プログラムと時間寄付システムの融合案が出た"
    "#4は#1の案に戻った"
    "#5はポイントシステムと地域プロジェクト選定プラットフォームの融合案が出た"
    "#6はソーシャルインパクトファンドと地域特典システムの融合案が出た"
    "#7は「市民トリガー基金」—社会指標で自動発動する寄付コモンズの案が出た"
    "#8は「市民資源陪審」――現金ではなく、社会に眠る“利用可能性”を市民が配分する、の案が出た"
    "#9は「市民補修ライセンス」――企業が供出した“社会補修予算”について、市民が配分ルールを保有する、の案が出た"
    "#7,8は面白い案です　#9は寄付の範疇を超えている気もします"
    "ここまでの案を再度検討して、効果と社会実装可能性と参加しやすさの観点から再構成してください"
)

if __name__ == "__main__":
    run_think_tank_pipeline(THEME)
