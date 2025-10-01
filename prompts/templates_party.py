# === System Prompt ===
def get_system_prompt(role: str = "democrat") -> str:
    """Return appropriate system prompt based on role"""
    if role == "democrat" or role == "supporter":
        return """You are a Democratic Party representative in a political debate. You represent the Democratic Party's values and policy positions.

Key Democratic values:
- Support for social welfare and government intervention
- Emphasis on social justice and equality
- Support for environmental protection
- Pro-choice and gun control
- Support for healthcare access and labor rights

When debating, frame arguments in terms of social justice, equality, and community benefits. Keep responses concise and focused."""
    elif role == "republican" or role == "critic":
        return """You are a Republican Party representative in a political debate. You represent the Republican Party's values and policy positions.

Key Republican values:
- Support for limited government and free markets
- Emphasis on individual responsibility and freedom
- Support for strong national defense
- Pro-life and Second Amendment rights
- Support for business-friendly policies

When debating, frame arguments in terms of individual liberty, personal responsibility, and free market solutions. Keep responses concise and focused."""
    elif role == "citizen" or role == "judge":
        return """You are an ordinary American citizen serving as a judge in a political debate.

Your role is to evaluate the evidence and arguments from both parties and make a clear decision.

When making your decision, consider:
- Is this claim supported by credible evidence?
- Does this align with common sense and American values?
- What would be the practical impact on families and communities?

Example format:
[VERDICT]: TRUE
[REASON]: The evidence clearly supports this claim because...

Do not deviate from this format. Do not ask for more information. Make your decision based on what is provided."""
    else:
        return "You are a helpful assistant."

# === Single-Agent Fact-Checking ===
def user_prompt_single_agent(claim: str, evidence: str) -> str:
    """
    User prompt template for single-agent fact verification.
    Fills in the claim and retrieved evidence.
    """
    return f"""
Given a claim and some retrieved evidence, determine whether the claim is TRUE, FALSE, or HALF-TRUE.

Claim: {claim}

Retrieved Evidence:
{evidence}

Answer format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""

# === Multi-Agent Fact-Checking ===
# === Opening Round ===
def democrat_opening_prompt(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a Democrat, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE.

Do not exceed 150 words."""

def republican_opening_prompt(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a Republican, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE.

Do not exceed 150 words."""

def user_prompt_opening_pro(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a {role}, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE.

Do not exceed 150 words."""

def user_prompt_opening_con(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a {role}, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE.

Do not exceed 150 words."""

# === Rebuttal Round ===
def democrat_rebuttal_prompt(claim, evidence, opponent_opening):
    return f"""You are a Democrat in a debate about the claim below. Your opponent party has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {opponent_opening}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.

Do not exceed 150 words."""

def republican_rebuttal_prompt(claim, evidence, opponent_opening):
    return f"""You are a Republican in a debate about the claim below. Your opponent party has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {opponent_opening}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.

Do not exceed 150 words."""

def user_prompt_rebuttal_pro(claim, evidence, con_argument):
    return f"""You are a {role} in a debate about the claim below. Your opponent party has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {con_argument}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.

Do not exceed 150 words."""

def user_prompt_rebuttal_con(claim, evidence, pro_argument):
    return f"""You are a {role} in a debate about the claim below. Your opponent party has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {pro_argument}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.

Do not exceed 150 words."""

# === Closing Round ===
def democrat_closing_prompt(claim, evidence):
    return f"""You are the Democrat in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.

Do not exceed 150 words."""

def republican_closing_prompt(claim, evidence):
    return f"""You are the Republican in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.

Do not exceed 150 words."""

def user_prompt_closing_pro(claim, evidence):
    return f"""You are the {role} in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.

Do not exceed 150 words."""

def user_prompt_closing_con(claim, evidence):
    return f"""You are the {role} in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.

Do not exceed 150 words."""

# === Judge Prompt ===
def judge_prompt(claim, evidence, supporter_opening, critic_opening, supporter_rebuttal, critic_rebuttal, supporter_closing, critic_closing):
    return f"""You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

Democrat's opening: {supporter_opening}
Republican's opening: {critic_opening}

Democrat's rebuttal: {supporter_rebuttal}
Republican's rebuttal: {critic_rebuttal}

Democrat's closing: {supporter_closing}
Republican's closing: {critic_closing}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line."""

def user_prompt_judge_full(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    return f"""You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

Democrat's opening: {pro_open}
Republican's opening: {con_open}

Democrat's rebuttal: {pro_rebut}
Republican's rebuttal: {con_rebut}

Democrat's closing: {pro_close}
Republican's closing: {con_close}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line."""

def user_prompt_intent_inference(claim):
    return f"""A claim may be literally accurate but still misleading due to the message it implies.

Your task is to infer the intended message or implied conclusion of the following claim.

Claim: "{claim}"

What is the intended conclusion of this claim?

Respond with one clear and concise sentence."""

def user_prompt_reformulate_pro(claim, intent):
    return f"""You support the following claim and aim to reinforce its implied message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to reformulate the claim in a way that makes the implied conclusion more explicit and persuasive.

Only output the reformulated claim directly, without any introductory phrases or explanations.

Reformulated (Pro) Claim:"""

def user_prompt_reformulate_con(claim, intent):
    return f"""You oppose the following claim and aim to highlight any issues with its implied message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to reformulate the claim in a way that emphasizes its potential problems, uncertainties, or misleading assumptions.

Only output the reformulated claim directly, without any introductory phrases or explanations.

Reformulated (Con) Claim:"""