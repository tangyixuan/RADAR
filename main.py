import argparse
import json
import os

from tqdm import tqdm

from model.loader import load_model


def run_single_agent(claim, evidence, model_info):
    from agents.single_agent import set_model_info, verify_claim

    set_model_info(model_info)
    return verify_claim(claim, evidence)


def run_multi_agent(claim, evidence, model_info):
    from agents.multi_agents import (
        set_model_info,
        opening_pro,
        rebuttal_pro,
        closing_pro,
        opening_con,
        rebuttal_con,
        closing_con,
        judge_final_verdict,
    )

    set_model_info(model_info)

    print("\n=== Running Multi-Agent Debate (3 rounds) ===")
    pro_open = opening_pro(claim, evidence)
    con_open = opening_con(claim, evidence)
    pro_rebut = rebuttal_pro(claim, evidence, con_open)
    con_rebut = rebuttal_con(claim, evidence, pro_open)
    pro_close = closing_pro(claim, evidence)
    con_close = closing_con(claim, evidence)
    final_result = judge_final_verdict(
        claim,
        evidence,
        pro_open,
        con_open,
        pro_rebut,
        con_rebut,
        pro_close,
        con_close,
    )
    return (
        pro_open,
        con_open,
        pro_rebut,
        con_rebut,
        pro_close,
        con_close,
        final_result,
    )


def run_multi_agent_people(claim, evidence, model_info):
    from agents.multi_agent_people import (
        set_model_info,
        opening_politician,
        rebuttal_politician,
        closing_politician,
        opening_scientist,
        rebuttal_scientist,
        closing_scientist,
        judge_final_verdict,
    )

    set_model_info(model_info)

    print("\n=== Running Multi-Agent People Debate (Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    final_result = judge_final_verdict(
        claim,
        evidence,
        pol_open,
        sci_open,
        pol_rebut,
        sci_rebut,
        pol_close,
        sci_close,
    )
    return (
        pol_open,
        sci_open,
        pol_rebut,
        sci_rebut,
        pol_close,
        sci_close,
        final_result,
    )


def run_multi_agent_people_continue_check(claim, evidence, model_info):
    from agents.multi_agent_people_continue_check import (
        set_model_info,
        run_multi_agent_people_continue_check as run_people_continue_check,
    )

    set_model_info(model_info)

    print("\n=== Running Multi-Agent People Debate (Continuation Judge) ===")
    return run_people_continue_check(claim, evidence)


def run_multi_agent_people_round_judges(claim, evidence, model_info):
    from agents.multi_agent_people_round_judges import (
        set_model_info,
        run_multi_agent_people_round_judges as run_people_round,
    )

    set_model_info(model_info)

    print("\n=== Running Multi-Agent People Debate with Round Judges ===")
    return run_people_round(claim, evidence)


def run_multi_agent_people_hybrid_adaptive(claim, evidence, model_info, tau_s, tau_v):
    from agents.multi_agent_people_hybrid import (
        set_model_info,
        run_multi_agent_people_hybrid_adaptive as run_people_hybrid_adaptive,
    )

    set_model_info(model_info)

    print(
        "\n=== Running Multi-Agent People Hybrid Debate with Adaptive Early Stopping "
        f"(tau_s={tau_s}, tau_v={tau_v}) ==="
    )
    return run_people_hybrid_adaptive(claim, evidence, tau_s, tau_v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "multi",
            "multi_people",
            "multi_people_continue_check",
            "multi_people_round_judges",
            "multi_people_hybrid_adaptive",
        ],
        default="single",
        help="Choose inference mode.",
    )
    parser.add_argument(
        "--model",
        choices=["llama", "qwen", "gpt"],
        default="llama",
        help="Choose model type: llama, qwen, or gpt",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/yirui/mad_debate/mad_formal/model/qwen2.5-7b-instruct",
        help="Path to local model (for llama or qwen)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key (required for gpt model)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing examples.",
    )
    parser.add_argument(
        "--tau_s",
        type=float,
        default=0.5,
        help="Stop margin threshold for adaptive early stopping (default: 0.5)",
    )
    parser.add_argument(
        "--tau_v",
        type=float,
        default=0.7,
        help="Veracity confidence threshold for adaptive early stopping (default: 0.7)",
    )
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    if args.model == "gpt":
        model_info = load_model(model_type=args.model, api_key=args.api_key)
    elif args.model == "qwen":
        model_path = args.model_path or "Qwen/Qwen2.5-7B-Instruct"
        model_info = load_model(model_path=model_path, model_type=args.model)
    else:
        model_path = args.model_path
        model_info = load_model(model_path=model_path, model_type=args.model)

    print(f"Model loaded successfully: {args.model}")

    print(f"Loading input file: {args.input_file}")
    with open(args.input_file, "r") as f:
        all_examples = json.load(f)

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file = os.path.join(
        "data", f"{input_basename}_answer_map_{args.mode}_{args.model}.json"
    )

    print(f"Output will be saved to: {output_file}")
    print(
        f"Processing {len(all_examples)} examples in {args.mode} mode with {args.model} model"
    )

    try:
        with open(output_file, "r") as f:
            answer_map = json.load(f)
        print(f"Found existing results file with {len(answer_map)} completed examples")
    except FileNotFoundError:
        answer_map = {}
        print("No existing results file found, starting from scratch...")

    for example_id, example in tqdm(
        all_examples.items(), desc=f"Processing examples ({args.mode} + {args.model})"
    ):
        if example_id in answer_map:
            continue

        claim = example["claim"]
        evidence = example["evidence_full_text"]

        if args.mode == "single":
            result = run_single_agent(claim, evidence, model_info)
            answer_map[example_id] = [result]

        elif args.mode == "multi":
            (
                pro_open,
                con_open,
                pro_rebut,
                con_rebut,
                pro_close,
                con_close,
                final_result,
            ) = run_multi_agent(claim, evidence, model_info)
            answer_map[example_id] = {
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result,
            }

        elif args.mode == "multi_people":
            (
                pol_open,
                sci_open,
                pol_rebut,
                sci_rebut,
                pol_close,
                sci_close,
                final_result,
            ) = run_multi_agent_people(claim, evidence, model_info)
            answer_map[example_id] = {
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result,
            }

        elif args.mode == "multi_people_continue_check":
            result = run_multi_agent_people_continue_check(
                claim, evidence, model_info
            )
            answer_map[example_id] = result

        elif args.mode == "multi_people_round_judges":
            result = run_multi_agent_people_round_judges(claim, evidence, model_info)
            answer_map[example_id] = result

        elif args.mode == "multi_people_hybrid_adaptive":
            result = run_multi_agent_people_hybrid_adaptive(
                claim,
                evidence,
                model_info,
                args.tau_s,
                args.tau_v,
            )
            answer_map[example_id] = result

        with open(output_file, "w") as f:
            json.dump(answer_map, f, indent=2)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(answer_map)} examples")


if __name__ == "__main__":
    main()
