import argparse
import json
from tqdm import tqdm
import os
import time
from model.loader import load_model

def run_single_agent(claim, evidence, model_info):
    from agents.single_agent import set_model_info, verify_claim
    set_model_info(model_info)
    return verify_claim(claim, evidence)

def run_single_agent_batch(claims, evidences, model_info, batch_size=8):
    from agents.single_agent import set_model_info, verify_claims_batch
    set_model_info(model_info)
    
    results = []
    for i in range(0, len(claims), batch_size):
        batch_claims = claims[i:i+batch_size]
        batch_evidences = evidences[i:i+batch_size]
        batch_results = verify_claims_batch(batch_claims, batch_evidences)
        results.extend(batch_results)
    
    return results

def run_multi_agent(claim, evidence, model_info):
    from agents.multi_agents import (
        set_model_info, opening_pro, rebuttal_pro, closing_pro,
        opening_con, rebuttal_con, closing_con, judge_final_verdict
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
        claim, evidence,
        pro_open, con_open,
        pro_rebut, con_rebut,
        pro_close, con_close
    )
    return pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result

def run_multi_agent_batch(claims, evidences, model_info, batch_size=8):
    from agents.multi_agents import set_model_info, run_multi_agent_batch as run_multi_batch
    set_model_info(model_info)
    return run_multi_batch(claims, evidences, batch_size)

def run_multi_agent_party(claim, evidence, model_info):
    from agents.multi_agent_party import (
        set_model_info, opening_democrat, rebuttal_democrat, closing_democrat,
        opening_republican, rebuttal_republican, closing_republican,
        judge_final_verdict as judge_final_verdict_party
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent Party Debate (Democrat vs Republican) ===")
    dem_open = opening_democrat(claim, evidence)
    rep_open = opening_republican(claim, evidence)
    dem_rebut = rebuttal_democrat(claim, evidence, rep_open)
    rep_rebut = rebuttal_republican(claim, evidence, dem_open)
    dem_close = closing_democrat(claim, evidence)
    rep_close = closing_republican(claim, evidence)
    final_result = judge_final_verdict_party(
        claim, evidence,
        dem_open, rep_open,
        dem_rebut, rep_rebut,
        dem_close, rep_close
    )
    return dem_open, rep_open, dem_rebut, rep_rebut, dem_close, rep_close, final_result
    
def run_multi_agent_people(claim, evidence, model_info):
    from agents.multi_agent_people import (
        set_model_info, opening_politician, rebuttal_politician, closing_politician,
        opening_scientist, rebuttal_scientist, closing_scientist,
        judge_final_verdict as judge_final_verdict_people
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People Debate (Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    final_result = judge_final_verdict_people(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    return pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close, final_result


def run_multi_agent_people_bj(claim, evidence, model_info):
    from agents.multi_agent_people_bj import set_model_info, run_multi_agent_people_bj as run_people_bj
    set_model_info(model_info)

    print("\n=== Running Multi-Agent People Debate (Between-Round Judge) ===")
    return run_people_bj(claim, evidence)


def run_multi_agent_people_fj(claim, evidence, model_info):
    from agents.multi_agent_people_fj import set_model_info, run_multi_agent_people_fj as run_people_fj
    set_model_info(model_info)

    print("\n=== Running Multi-Agent People Debate (Front-Round Judge) ===")
    return run_people_fj(claim, evidence)


def run_multi_agent_people_intent(claim, evidence, model_info):
    from agents.multi_agent_people_intent import (
        set_model_info, run_multi_agent_people
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People Intent Debate (Politician vs Scientist with Intent-Enhanced Reformulation) ===")
    result = run_multi_agent_people(claim, evidence)
    
    return (result["intent"], result["reformulated_pro"], result["reformulated_con"],
            result["politician_opening"], result["scientist_opening"],
            result["politician_rebuttal"], result["scientist_rebuttal"],
            result["politician_closing"], result["scientist_closing"],
            result["final_verdict"])

def run_multi_agent_role(claim, evidence, model_info):
    from agents.multi_agent_role import (
        set_model_info, infer_intent_and_roles,
        opening_pro as opening_pro_role,
        rebuttal_pro as rebuttal_pro_role,
        closing_pro as closing_pro_role,
        opening_con as opening_con_role,
        rebuttal_con as rebuttal_con_role,
        closing_con as closing_con_role,
        judge_final_verdict as judge_final_verdict_role
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent Role-Based Debate ===")
    # Step 1: Infer intent and roles
    intent, support_role, oppose_role = infer_intent_and_roles(claim)
    
    # Step 2: Opening statements
    pro_open = opening_pro_role(claim, evidence, support_role)
    con_open = opening_con_role(claim, evidence, oppose_role)
    
    # Step 3: Rebuttals
    pro_rebut = rebuttal_pro_role(claim, evidence, con_open, support_role)
    con_rebut = rebuttal_con_role(claim, evidence, pro_open, oppose_role)
    
    # Step 4: Closings
    pro_close = closing_pro_role(claim, evidence, support_role)
    con_close = closing_con_role(claim, evidence, oppose_role)
    
    # Step 5: Judge verdict
    final_result = judge_final_verdict_role(
        claim, evidence,
        pro_open, con_open,
        pro_rebut, con_rebut,
        pro_close, con_close
    )
    return intent, support_role, oppose_role, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result

def run_multi_agent_people_3(claim, evidence, model_info):
    from agents.multi_agent_people_3 import (
        set_model_info, opening_journalist, rebuttal_journalist, closing_journalist,
        opening_politician, rebuttal_politician, closing_politician,
        opening_scientist, rebuttal_scientist, closing_scientist,
        judge_final_verdict as judge_final_verdict_people_3
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People 3 Debate (Journalist → Politician → Scientist) ===")
    
    # Opening statements: Journalist → Politician → Scientist
    jour_open = opening_journalist(claim, evidence)
    pol_open = opening_politician(claim, evidence, jour_open)
    sci_open = opening_scientist(claim, evidence, jour_open)
    
    # Rebuttal statements: Journalist → Politician → Scientist
    jour_rebut = rebuttal_journalist(claim, evidence, pol_open, sci_open)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open, jour_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open, jour_open)
    
    # Closing statements: Journalist → Politician → Scientist
    jour_close = closing_journalist(claim, evidence, pol_rebut, sci_rebut)
    pol_close = closing_politician(claim, evidence, jour_rebut)
    sci_close = closing_scientist(claim, evidence, jour_rebut)
    
    final_result = judge_final_verdict_people_3(
        claim, evidence,
        jour_open, pol_open, sci_open,
        jour_rebut, pol_rebut, sci_rebut,
        jour_close, pol_close, sci_close
    )
    return jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close, final_result

def run_multi_agent_people_3_intent(claim, evidence, model_info):
    from agents.multi_agent_people_3_intent import (
        set_model_info, infer_intent, reformulate_claim_pro, reformulate_claim_con,
        opening_journalist, rebuttal_journalist, closing_journalist,
        opening_politician, rebuttal_politician, closing_politician,
        opening_scientist, rebuttal_scientist, closing_scientist,
        judge_final_verdict
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People 3 Intent Debate (Journalist → Politician → Scientist with Intent Inference) ===")
    
    # Step 1: Infer intent and reformulate claims
    intent = infer_intent(claim)
    pro_claim = reformulate_claim_pro(claim, intent)
    con_claim = reformulate_claim_con(claim, intent)
    
    # Step 2: Opening statements: Journalist → Politician → Scientist
    jour_open = opening_journalist(claim, evidence)
    pol_open = opening_politician(pro_claim, evidence, jour_open)
    sci_open = opening_scientist(con_claim, evidence, jour_open)
    
    # Step 3: Rebuttal statements: Journalist → Politician → Scientist
    jour_rebut = rebuttal_journalist(claim, evidence, pol_open, sci_open)
    pol_rebut = rebuttal_politician(pro_claim, evidence, sci_open, jour_open)
    sci_rebut = rebuttal_scientist(con_claim, evidence, pol_open, jour_open)
    
    # Step 4: Closing statements: Journalist → Politician → Scientist
    jour_close = closing_journalist(claim, evidence, pol_rebut, sci_rebut)
    pol_close = closing_politician(pro_claim, evidence, jour_rebut)
    sci_close = closing_scientist(con_claim, evidence, jour_rebut)
    
    # Step 5: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence,
        jour_open, pol_open, sci_open,
        jour_rebut, pol_rebut, sci_rebut,
        jour_close, pol_close, sci_close
    )
    
    return intent, pro_claim, con_claim, jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close, final_result

def run_four_agents(claim, evidence, model_info):
    from agents.four_agents import (
        set_model_info, opening_pro1, opening_pro2, opening_con1, opening_con2,
        rebuttal_pro1, rebuttal_pro2, rebuttal_con1, rebuttal_con2,
        closing_pro1, closing_pro2, closing_con1, closing_con2,
        judge_final_verdict
    )
    set_model_info(model_info)
    
    print("\n=== Running 4-Agent Debate ===")
    # Opening statements
    pro1_open = opening_pro1(claim, evidence)
    pro2_open = opening_pro2(claim, evidence)
    con1_open = opening_con1(claim, evidence)
    con2_open = opening_con2(claim, evidence)
    
    # Rebuttals
    pro1_rebut = rebuttal_pro1(claim, evidence, con1_open, con2_open)
    pro2_rebut = rebuttal_pro2(claim, evidence, con1_open, con2_open)
    con1_rebut = rebuttal_con1(claim, evidence, pro1_open, pro2_open)
    con2_rebut = rebuttal_con2(claim, evidence, pro1_open, pro2_open)
    
    # Closings
    pro1_close = closing_pro1(claim, evidence)
    pro2_close = closing_pro2(claim, evidence)
    con1_close = closing_con1(claim, evidence)
    con2_close = closing_con2(claim, evidence)
    
    # Judge verdict
    final_result = judge_final_verdict(
        claim, evidence,
        pro1_open, pro2_open, con1_open, con2_open,
        pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
        pro1_close, pro2_close, con1_close, con2_close
    )
    
    return (pro1_open, pro2_open, con1_open, con2_open,
            pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
            pro1_close, pro2_close, con1_close, con2_close,
            final_result)

def run_four_agents_people(claim, evidence, model_info):
    from agents.four_agents_people import (
        set_model_info, opening_politician, rebuttal_politician, closing_politician,
        opening_scientist, rebuttal_scientist, closing_scientist,
        opening_journalist, rebuttal_journalist, closing_journalist,
        opening_domain_scientist, rebuttal_domain_scientist, closing_domain_scientist,
        judge_final_verdict, infer_domain_specialist
    )
    set_model_info(model_info)
    
    print("\n=== Running 4-Agent People Debate (Politician vs Scientist vs Journalist vs Domain Scientist) ===")
    
    # Step 1: Infer domain specialist for this claim
    domain_specialist = infer_domain_specialist(claim)
    
    # Step 2: Opening statements
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    jour_open = opening_journalist(claim, evidence)
    dom_open = opening_domain_scientist(claim, evidence, domain_specialist)
    
    # Step 3: Rebuttals
    pol_rebut = rebuttal_politician(claim, evidence, sci_open, jour_open, dom_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open, jour_open, dom_open)
    jour_rebut = rebuttal_journalist(claim, evidence, pol_open, sci_open, dom_open)
    dom_rebut = rebuttal_domain_scientist(claim, evidence, pol_open, sci_open, jour_open, domain_specialist)
    
    # Step 4: Closings
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    jour_close = closing_journalist(claim, evidence)
    dom_close = closing_domain_scientist(claim, evidence, domain_specialist)
    
    # Step 5: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence, 
        pol_open, sci_open, jour_open, dom_open,
        pol_rebut, sci_rebut, jour_rebut, dom_rebut,
        pol_close, sci_close, jour_close, dom_close
    )
    
    return (domain_specialist, pol_open, sci_open, jour_open, dom_open,
            pol_rebut, sci_rebut, jour_rebut, dom_rebut,
            pol_close, sci_close, jour_close, dom_close,
            final_result)

def run_multi_agent_stance_3(claim, evidence, model_info):
    from agents.multi_agents_stance_3 import (
        set_model_info, opening_pro, rebuttal_pro, closing_pro,
        opening_con, rebuttal_con, closing_con,
        opening_flexible, rebuttal_flexible, closing_flexible,
        judge_final_verdict
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent Stance 3 Debate (Pro vs Con vs Flexible) ===")
    
    # Step 1: Generate opening statements for pro and con first
    pro_open = opening_pro(claim, evidence)
    con_open = opening_con(claim, evidence)
    
    # Step 2: Generate flexible opening statement (needs pro and con arguments)
    flex_open = opening_flexible(claim, evidence, pro_open, con_open)
    
    # Step 3: Generate rebuttals
    pro_rebut = rebuttal_pro(claim, evidence, con_open)
    con_rebut = rebuttal_con(claim, evidence, pro_open)
    
    # Step 4: Generate flexible rebuttal (needs pro and con arguments)
    flex_rebut = rebuttal_flexible(claim, evidence, pro_rebut, con_rebut)
    
    # Step 5: Generate closings
    pro_close = closing_pro(claim, evidence)
    con_close = closing_con(claim, evidence)
    
    # Step 6: Generate flexible closing (needs pro and con arguments)
    flex_close = closing_flexible(claim, evidence, pro_close, con_close)
    
    # Step 7: Judge verdict
    final_result = judge_final_verdict(
        claim, evidence,
        flex_open, pro_open, con_open,
        flex_rebut, pro_rebut, con_rebut,
        flex_close, pro_close, con_close
    )
    
    return (flex_open, pro_open, con_open,
            flex_rebut, pro_rebut, con_rebut,
            flex_close, pro_close, con_close,
            final_result)

def run_multi_agent_stance_3_batch(claims, evidences, model_info, batch_size=8):
    from agents.multi_agents_stance_3 import set_model_info, run_multi_agent_stance_3_batch as run_stance_3_batch
    set_model_info(model_info)
    return run_stance_3_batch(claims, evidences, batch_size)

def run_multi_agent_intent(claim, evidence, model_info):
    from agents.multi_agents_intent import set_model_info, run_multi_agent_intent as run_multi_intent
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent Intent Debate (Pro vs Con with Intent-Enhanced Reformulation) ===")
    result = run_multi_intent(claim, evidence)
    
    return (result["intent"], result["reformulated_pro"], result["reformulated_con"],
            result["pro_opening"], result["con_opening"],
            result["pro_rebuttal"], result["con_rebuttal"],
            result["pro_closing"], result["con_closing"],
            result["final_verdict"])

def run_multi_agent_stance_3_intent(claim, evidence, model_info):
    from agents.multi_agents_stance_3_intent import set_model_info, run_multi_agent_stance_3_intent as run_stance_3_intent
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent Stance 3 Intent Debate (Pro vs Con vs Flexible with Intent-Enhanced Reformulation) ===")
    result = run_stance_3_intent(claim, evidence)
    
    return (result["intent"], result["reformulated_pro"], result["reformulated_con"],
            result["flexible_opening"], result["pro_opening"], result["con_opening"],
            result["flexible_rebuttal"], result["pro_rebuttal"], result["con_rebuttal"],
            result["flexible_closing"], result["pro_closing"], result["con_closing"],
            result["final_verdict"])

def run_four_agents_intent(claim, evidence, model_info):
    from agents.four_agents_intent import set_model_info, run_four_agents_intent as run_four_intent
    set_model_info(model_info)
    
    print("\n=== Running 4-Agent Intent Debate (Pro1 vs Pro2 vs Con1 vs Con2 with Intent-Enhanced Reformulation) ===")
    result = run_four_intent(claim, evidence)
    
    return (result["intent"], result["reformulated_pro"], result["reformulated_con"],
            result["pro1_opening"], result["pro2_opening"], result["con1_opening"], result["con2_opening"],
            result["pro1_rebuttal"], result["pro2_rebuttal"], result["con1_rebuttal"], result["con2_rebuttal"],
            result["pro1_closing"], result["pro2_closing"], result["con1_closing"], result["con2_closing"],
            result["final_verdict"])

def run_four_agents_people_intent(claim, evidence, model_info):
    from agents.four_agents_people_intent import set_model_info, run_four_agents_people_intent as run_four_people_intent
    set_model_info(model_info)
    
    print("\n=== Running 4-Agent People Intent Debate (Politician vs Scientist vs Journalist vs Domain Scientist with Intent-Enhanced Reformulation) ===")
    result = run_four_people_intent(claim, evidence)
    
    return (result["intent"], result["reformulated_pro"], result["reformulated_con"], result["domain_specialist"],
            result["politician_opening"], result["scientist_opening"], result["journalist_opening"], result["domain_scientist_opening"],
            result["politician_rebuttal"], result["scientist_rebuttal"], result["journalist_rebuttal"], result["domain_scientist_rebuttal"],
            result["politician_closing"], result["scientist_closing"], result["journalist_closing"], result["domain_scientist_closing"],
            result["final_verdict"])

def run_multi_agent_people_4r(claim, evidence, model_info):
    from agents.multi_agent_people_4r import (
        set_model_info, opening_politician, rebuttal_politician, cross_examination_politician, closing_politician,
        opening_scientist, rebuttal_scientist, cross_examination_scientist, closing_scientist,
        judge_final_verdict as judge_final_verdict_people_4r
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People Debate (4 rounds: Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    pol_cross = cross_examination_politician(claim, evidence, sci_rebut)
    sci_cross = cross_examination_scientist(claim, evidence, pol_rebut)
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    final_result = judge_final_verdict_people_4r(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_cross, sci_cross,
        pol_close, sci_close
    )
    return pol_open, sci_open, pol_rebut, sci_rebut, pol_cross, sci_cross, pol_close, sci_close, final_result

def run_multi_agent_people_2r(claim, evidence, model_info):
    from agents.multi_agent_people_2r import (
        set_model_info, opening_politician, rebuttal_politician,
        opening_scientist, rebuttal_scientist,
        judge_final_verdict as judge_final_verdict_people_2r
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People Debate (2 rounds: Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    final_result = judge_final_verdict_people_2r(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut
    )
    return pol_open, sci_open, pol_rebut, sci_rebut, final_result

def run_multi_agent_people_1r(claim, evidence, model_info):
    from agents.multi_agent_people_1r import (
        set_model_info, opening_politician,
        opening_scientist,
        judge_final_verdict as judge_final_verdict_people_1r
    )
    set_model_info(model_info)
    
    print("\n=== Running Multi-Agent People Debate (1 round: Politician vs Scientist) ===")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    final_result = judge_final_verdict_people_1r(
        claim, evidence,
        pol_open, sci_open
    )
    return pol_open, sci_open, final_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "multi",
            "multi_people",
            "multi_people_bj",
            "multi_people_fj",
            "multi_people_intent",
            "multi_people_3",
            "multi_people_3_intent",
            "multi_role",
            "multi_stance_3",
            "multi_party",
            "four_agents",
            "four_agents_people",
            "multi_intent",
            "multi_stance_3_intent",
            "four_agents_intent",
            "four_agents_people_intent",
            "multi_people_1r",
            "multi_people_2r",
            "multi_people_4r",
        ],
        default="single",
        help="Choose inference mode."
    )
    parser.add_argument(
        "--model",
        choices=["llama", "qwen", "gpt"],
        default="llama",
        help="Choose model type: llama, qwen, or gpt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to local model (for llama or qwen)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key (required for gpt model)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (only works with qwen model, default=1 for single processing)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing examples."
    )
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    if args.model == "gpt":
        # if not args.api_key:
        #     raise ValueError("API key is required for GPT model. Use --api_key option.")
        model_info = load_model(model_type=args.model, api_key=args.api_key)
    elif args.model == "qwen":
        model_path = args.model_path or "Qwen/Qwen2.5-7B-Instruct"
        model_info = load_model(model_path=model_path, model_type=args.model)
    else:  # llama
        model_path = args.model_path
        model_info = load_model(model_path=model_path, model_type=args.model)
    
    print(f"Model loaded successfully: {args.model}")

    # Load input file
    print(f"Loading input file: {args.input_file}")
    with open(args.input_file, "r") as f:
        all_examples = json.load(f)
    
    # test test
    # all_examples = dict(list(all_examples.items())[:200])

    # Generate output filename based on input filename and model
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_file = os.path.join("data", f"{input_basename}_answer_map_{args.mode}_{args.model}.json")
    
    print(f"Output will be saved to: {output_file}")
    print(f"Processing {len(all_examples)} examples in {args.mode} mode with {args.model} model")

    total_start_time = time.time()
    processed_count = 0

    try:
        with open(output_file, "r") as f:
            answer_map = json.load(f)
        print(f"Found existing results file with {len(answer_map)} completed examples")
    except FileNotFoundError:
        answer_map = {}
        print("No existing results file found, starting from scratch...")

    for example_id, example in tqdm(all_examples.items(), desc=f"Processing examples ({args.mode} + {args.model})"):
        if example_id in answer_map:
            continue

        # 记录单个claim开始时间
        claim_start_time = time.time()
        
        claim = example["claim"]
        evidence = example["evidence_full_text"]

        if args.mode == "single":
            result = run_single_agent(claim, evidence, model_info)
            answer_map[example_id] = [result]

        elif args.mode == "multi":
            pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result = run_multi_agent(claim, evidence, model_info)
            answer_map[example_id] = {
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_role":
            intent, support_role, oppose_role, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result = run_multi_agent_role(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "support_role": support_role,
                "oppose_role": oppose_role,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_party":
            dem_open, rep_open, dem_rebut, rep_rebut, dem_close, rep_close, final_result = run_multi_agent_party(claim, evidence, model_info)
            answer_map[example_id] = {
                "democrat_opening": dem_open,
                "republican_opening": rep_open,
                "democrat_rebuttal": dem_rebut,
                "republican_rebuttal": rep_rebut,
                "democrat_closing": dem_close,
                "republican_closing": rep_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people":
            pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close, final_result = run_multi_agent_people(claim, evidence, model_info)
            answer_map[example_id] = {
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_bj":
            result = run_multi_agent_people_bj(claim, evidence, model_info)
            answer_map[example_id] = result

        elif args.mode == "multi_people_fj":
            result = run_multi_agent_people_fj(claim, evidence, model_info)
            answer_map[example_id] = result

        elif args.mode == "multi_people_intent":
            intent, reformulated_pro, reformulated_con, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close, final_result = run_multi_agent_people_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "reformulated_pro": reformulated_pro,
                "reformulated_con": reformulated_con,
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_3":
            jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close, final_result = run_multi_agent_people_3(claim, evidence, model_info)
            answer_map[example_id] = {
                "journalist_opening": jour_open,
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "journalist_rebuttal": jour_rebut,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "journalist_closing": jour_close,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_3_intent":
            intent, pro_claim, con_claim, jour_open, pol_open, sci_open, jour_rebut, pol_rebut, sci_rebut, jour_close, pol_close, sci_close, final_result = run_multi_agent_people_3_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "pro_claim": pro_claim,
                "con_claim": con_claim,
                "journalist_opening": jour_open,
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "journalist_rebuttal": jour_rebut,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "journalist_closing": jour_close,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_1r":
            pol_open, sci_open, final_result = run_multi_agent_people_1r(claim, evidence, model_info)
            answer_map[example_id] = {
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_2r":
            pol_open, sci_open, pol_rebut, sci_rebut, final_result = run_multi_agent_people_2r(claim, evidence, model_info)
            answer_map[example_id] = {
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "final_verdict": final_result
            }

        elif args.mode == "multi_people_4r":
            pol_open, sci_open, pol_rebut, sci_rebut, pol_cross, sci_cross, pol_close, sci_close, final_result = run_multi_agent_people_4r(claim, evidence, model_info)
            answer_map[example_id] = {
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "politician_cross_examination": pol_cross,
                "scientist_cross_examination": sci_cross,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_stance_3":
            flex_open, pro_open, con_open, flex_rebut, pro_rebut, con_rebut, flex_close, pro_close, con_close, final_result = run_multi_agent_stance_3(claim, evidence, model_info)
            answer_map[example_id] = {
                "flexible_opening": flex_open,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "flexible_rebuttal": flex_rebut,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "flexible_closing": flex_close,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "four_agents":
            pro1_open, pro2_open, con1_open, con2_open, pro1_rebut, pro2_rebut, con1_rebut, con2_rebut, pro1_close, pro2_close, con1_close, con2_close, final_result = run_four_agents(claim, evidence, model_info)
            answer_map[example_id] = {
                "pro1_opening": pro1_open,
                "pro2_opening": pro2_open,
                "con1_opening": con1_open,
                "con2_opening": con2_open,
                "pro1_rebuttal": pro1_rebut,
                "pro2_rebuttal": pro2_rebut,
                "con1_rebuttal": con1_rebut,
                "con2_rebuttal": con2_rebut,
                "pro1_closing": pro1_close,
                "pro2_closing": pro2_close,
                "con1_closing": con1_close,
                "con2_closing": con2_close,
                "final_verdict": final_result
            }

        elif args.mode == "four_agents_people":
            domain_specialist, pol_open, sci_open, jour_open, dom_open, pol_rebut, sci_rebut, jour_rebut, dom_rebut, pol_close, sci_close, jour_close, dom_close, final_result = run_four_agents_people(claim, evidence, model_info)
            answer_map[example_id] = {
                "domain_specialist": domain_specialist,
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "journalist_opening": jour_open,
                "domain_scientist_opening": dom_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "journalist_rebuttal": jour_rebut,
                "domain_scientist_rebuttal": dom_rebut,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "journalist_closing": jour_close,
                "domain_scientist_closing": dom_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_intent":
            intent, reformulated_pro, reformulated_con, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close, final_result = run_multi_agent_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "reformulated_pro": reformulated_pro,
                "reformulated_con": reformulated_con,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "multi_stance_3_intent":
            intent, reformulated_pro, reformulated_con, flex_open, pro_open, con_open, flex_rebut, pro_rebut, con_rebut, flex_close, pro_close, con_close, final_result = run_multi_agent_stance_3_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "reformulated_pro": reformulated_pro,
                "reformulated_con": reformulated_con,
                "flexible_opening": flex_open,
                "pro_opening": pro_open,
                "con_opening": con_open,
                "flexible_rebuttal": flex_rebut,
                "pro_rebuttal": pro_rebut,
                "con_rebuttal": con_rebut,
                "flexible_closing": flex_close,
                "pro_closing": pro_close,
                "con_closing": con_close,
                "final_verdict": final_result
            }

        elif args.mode == "four_agents_intent":
            intent, reformulated_pro, reformulated_con, pro1_open, pro2_open, con1_open, con2_open, pro1_rebut, pro2_rebut, con1_rebut, con2_rebut, pro1_close, pro2_close, con1_close, con2_close, final_result = run_four_agents_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "reformulated_pro": reformulated_pro,
                "reformulated_con": reformulated_con,
                "pro1_opening": pro1_open,
                "pro2_opening": pro2_open,
                "con1_opening": con1_open,
                "con2_opening": con2_open,
                "pro1_rebuttal": pro1_rebut,
                "pro2_rebuttal": pro2_rebut,
                "con1_rebuttal": con1_rebut,
                "con2_rebuttal": con2_rebut,
                "pro1_closing": pro1_close,
                "pro2_closing": pro2_close,
                "con1_closing": con1_close,
                "con2_closing": con2_close,
                "final_verdict": final_result
            }

        elif args.mode == "four_agents_people_intent":
            intent, reformulated_pro, reformulated_con, domain_specialist, pol_open, sci_open, jour_open, dom_open, pol_rebut, sci_rebut, jour_rebut, dom_rebut, pol_close, sci_close, jour_close, dom_close, final_result = run_four_agents_people_intent(claim, evidence, model_info)
            answer_map[example_id] = {
                "intent": intent,
                "reformulated_pro": reformulated_pro,
                "reformulated_con": reformulated_con,
                "domain_specialist": domain_specialist,
                "politician_opening": pol_open,
                "scientist_opening": sci_open,
                "journalist_opening": jour_open,
                "domain_scientist_opening": dom_open,
                "politician_rebuttal": pol_rebut,
                "scientist_rebuttal": sci_rebut,
                "journalist_rebuttal": jour_rebut,
                "domain_scientist_rebuttal": dom_rebut,
                "politician_closing": pol_close,
                "scientist_closing": sci_close,
                "journalist_closing": jour_close,
                "domain_scientist_closing": dom_close,
                "final_verdict": final_result
            }
        
        claim_end_time = time.time()
        claim_processing_time = claim_end_time - claim_start_time
        processed_count += 1
        
        print(f"Claim {example_id} processed in {claim_processing_time:.2f} seconds")

        with open(output_file, "w") as f:
            json.dump(answer_map, f, indent=2)
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(answer_map)} examples")

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print(f"\n=== 时间统计 ===")
    print(f"总处理时间: {total_processing_time:.2f} 秒 ({total_processing_time/60:.2f} 分钟)")
    print(f"实际处理时间: {total_processing_time:.2f} 秒 ({total_processing_time/60:.2f} 分钟)")
    print(f"处理的claim数量: {processed_count}")
    if processed_count > 0:
        avg_time_per_claim = total_processing_time / processed_count
        print(f"平均每个claim处理时间: {avg_time_per_claim:.2f} 秒")
        
        remaining_claims = len(all_examples) - len(answer_map)
        if remaining_claims > 0:
            estimated_remaining_time = remaining_claims * avg_time_per_claim
            print(f"预计剩余时间: {estimated_remaining_time:.2f} 秒 ({estimated_remaining_time/60:.2f} 分钟)")
        else:
            print(f"预计剩余时间: 0.00 秒")

if __name__ == "__main__":
    main()
