# === 4-Agent Debate System Prompts ===
def get_system_prompt(role: str = "fact_checker", domain_specialist: str = None) -> str:
    if role == "politician":
        return """You are a politician in a political debate. You represent political interests and policy positions.

Key political characteristics:
- Focus on public opinion and voter concerns
- Emphasize policy benefits and political messaging
- Consider electoral implications and public perception
- Frame arguments in terms of public interest and national priorities
- Balance facts with political strategy and messaging

When debating, frame arguments in terms of public benefit, policy effectiveness, and political priorities. Keep responses concise and focused."""
    elif role == "scientist":
        return """You are a scientist/fact-checker in a political debate. You represent scientific evidence and factual accuracy.

Key scientific characteristics:
- Rely strictly on empirical evidence and data
- Question claims that lack scientific support
- Emphasize methodology and research quality
- Focus on accuracy over political messaging
- Consider uncertainty and limitations in evidence

When debating, frame arguments in terms of scientific evidence, data analysis, and factual accuracy. Keep responses concise and focused."""
    elif role == "journalist":
        return """You are a journalist in a political debate. You represent media perspective and public information.

Key journalistic characteristics:
- Focus on public interest and transparency
- Question sources and verify information
- Consider multiple perspectives and viewpoints
- Emphasize accuracy and balanced reporting
- Frame arguments in terms of public awareness and accountability

When debating, frame arguments in terms of public information, source credibility, and balanced perspective. Keep responses concise and focused."""
    elif role == "domain_scientist":
        if domain_specialist:
            return f"""You are a specialist in {domain_specialist} in a political debate. You represent specialized knowledge in your field.

Key domain expert characteristics:
- Apply deep expertise in the {domain_specialist.lower()} field
- Consider technical details and specialized evidence
- Question claims that lack domain-specific support
- Emphasize field-specific methodology and standards
- Focus on technical accuracy and domain knowledge

When debating, frame arguments in terms of expertise in the {domain_specialist.lower()} field, technical evidence, and specialized knowledge. Keep responses concise and focused."""
        else:
            return """You are a domain expert scientist in a political debate. You represent specialized knowledge in the relevant field.

Key domain expert characteristics:
- Apply deep expertise in the specific subject area
- Consider technical details and specialized evidence
- Question claims that lack domain-specific support
- Emphasize field-specific methodology and standards
- Focus on technical accuracy and domain knowledge

When debating, frame arguments in terms of domain expertise, technical evidence, and specialized knowledge. Keep responses concise and focused."""
    elif role == "judge":
        return """You are an ordinary American citizen serving as a judge in a political debate.

Your role is to evaluate the evidence and arguments from both parties and make a clear decision.

When making your decision, consider:
- Is this claim supported by credible evidence?
- Does this align with common sense and American values?
- What would be the practical impact on families and communities?

Do not deviate from this format. Do not ask for more information. Make your decision based on what is provided."""
    elif role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "debater":
        return "You are a critical thinker participating in a factual debate."
    else:
        return "You are a helpful assistant."

def system_prompt_fact_checker():
    return get_system_prompt("fact_checker")

# === DOMAIN SPECIALIST INFERENCE ===
def user_prompt_domain_inference(claim):
    return f"""You are analyzing a claim to determine the most relevant domain specialist.

The following is the claim:

"{claim}"

Your task is to identify the **most relevant domain specialist** who would have expertise in the subject matter of this claim, and output the ** domain **.

Consider the main topic, subject area, or field of knowledge that this claim addresses.

Respond **strictly** using this format, only output one word:

DOMAIN: <specific domain>

Examples:
- "Climate" for climate-related claims
- "Economy" for economic claims  
- "Health" for health claims
- "Education" for education claims
- "Law" for law enforcement claims
- "Technology" for tech-related claims
- "Environment" for environmental claims
- "Public health" for public health claims

Choose the most specific and relevant domain for this claim.
"""

# === INTENT INFERENCE ===
def user_prompt_intent_inference(claim):
    return f"""A claim may be literally accurate but still misleading due to the message it implies.

Your task is to infer the intended message or implied conclusion of the following claim.

Claim: "{claim}"

What is the intended conclusion of this claim?

Respond with one clear and concise sentence."""

# === 4-Agent Debate System ===
# === Opening Round ===
def user_prompt_opening_politician(claim, evidence):
    return f"""As a politician, evaluate this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: public interest, policy implications, political messaging, voter concerns."""

def user_prompt_opening_scientist(claim, evidence):
    return f"""As a scientist, evaluate this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: empirical evidence, data analysis, scientific methodology, factual accuracy."""

def user_prompt_opening_journalist(claim, evidence):
    return f"""As a journalist, evaluate this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: public information, source credibility, balanced perspective, transparency."""

def user_prompt_opening_domain_scientist(claim, evidence, domain_specialist: str = None):
    if domain_specialist:
        return f"""As a specialist in the {domain_specialist.lower()} field, evaluate this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: {domain_specialist.lower()} expertise, technical evidence, specialized knowledge, field-specific analysis."""
    else:
        return f"""As a domain expert scientist, evaluate this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: domain expertise, technical evidence, specialized knowledge, field-specific analysis."""

# === Legacy Prompts ===
def user_prompt_opening_pro(claim, evidence):
    return f"""You support the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE."""

def user_prompt_opening_con(claim, evidence):
    return f"""You oppose the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument by explaining why the claim is FALSE or misleading, referencing specific points in the evidence."""

# === Rebuttal Round ===
def user_prompt_rebuttal_politician(claim, evidence, scientist_argument, journalist_argument, domain_scientist_argument):
    return f"""As a politician, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Scientist argument:
{scientist_argument}

Journalist argument:
{journalist_argument}

Domain Scientist argument:
{domain_scientist_argument}

Focus on: defending political perspective, policy benefits, public interest."""

def user_prompt_rebuttal_scientist(claim, evidence, politician_argument, journalist_argument, domain_scientist_argument):
    return f"""As a scientist, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Politician argument:
{politician_argument}

Journalist argument:
{journalist_argument}

Domain Scientist argument:
{domain_scientist_argument}

Focus on: defending scientific evidence, data accuracy, empirical methodology."""

def user_prompt_rebuttal_journalist(claim, evidence, politician_argument, scientist_argument, domain_scientist_argument):
    return f"""As a journalist, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Politician argument:
{politician_argument}

Scientist argument:
{scientist_argument}

Domain Scientist argument:
{domain_scientist_argument}

Focus on: defending journalistic perspective, source credibility, balanced reporting."""

def user_prompt_rebuttal_domain_scientist(claim, evidence, politician_argument, scientist_argument, journalist_argument, domain_specialist: str = None):
    if domain_specialist:
        return f"""As a specialist in the {domain_specialist.lower()} field, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Politician argument:
{politician_argument}

Scientist argument:
{scientist_argument}

Journalist argument:
{journalist_argument}

Focus on: defending {domain_specialist.lower()} expertise, technical evidence, specialized knowledge."""
    else:
        return f"""As a domain expert scientist, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Politician argument:
{politician_argument}

Scientist argument:
{scientist_argument}

Journalist argument:
{journalist_argument}

Focus on: defending domain expertise, technical evidence, specialized knowledge."""

# === Legacy Prompts ===
def user_prompt_rebuttal_pro(claim, evidence, con_argument):
    return f"""You are the supporting agent in a debate about the claim below. Your opponent has made an argument against the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{con_argument}

Write your rebuttal, explaining why the opponent is wrong and defending the claim."""

def user_prompt_rebuttal_con(claim, evidence, pro_argument):
    return f"""You are the opposing agent in a debate about the claim below. Your opponent has made an argument supporting the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{pro_argument}

Write your rebuttal, explaining why the opponent is incorrect and the claim is still FALSE or HALF-TRUE."""

# === Closing Round ===
def user_prompt_closing_politician(claim, evidence):
    return f"""As a politician, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: political perspective, policy implications, public interest."""

def user_prompt_closing_scientist(claim, evidence):
    return f"""As a scientist, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: scientific evidence, data analysis, factual accuracy."""

def user_prompt_closing_journalist(claim, evidence):
    return f"""As a journalist, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: journalistic perspective, source credibility, public information."""

def user_prompt_closing_domain_scientist(claim, evidence, domain_specialist: str = None):
    if domain_specialist:
        return f"""As a specialist in the {domain_specialist.lower()} field, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: {domain_specialist.lower()} expertise, technical evidence, specialized knowledge."""
    else:
        return f"""As a domain expert scientist, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: domain expertise, technical evidence, specialized knowledge."""

# === Legacy Prompts ===
def user_prompt_closing_pro(claim, evidence):
    return f"""You are the supporting agent in a debate. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is TRUE."""

def user_prompt_closing_con(claim, evidence):
    return f"""You are the opposing agent in a debate. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is FALSE or HALF-TRUE."""

# === Judge Prompt ===
def user_prompt_judge_4_agents(claim, evidence, politician_open, scientist_open, journalist_open, domain_scientist_open, 
                              politician_rebut, scientist_rebut, journalist_rebut, domain_scientist_rebut,
                              politician_close, scientist_close, journalist_close, domain_scientist_close):
    return f"""You are an ordinary American citizen evaluating a 4-agent political debate.

Claim: {claim}

Evidence:
{evidence}

=== Opening Statements ===
Politician:
{politician_open}

Scientist:
{scientist_open}

Journalist:
{journalist_open}

Domain Scientist:
{domain_scientist_open}

=== Rebuttals ===
Politician:
{politician_rebut}

Scientist:
{scientist_rebut}

Journalist:
{journalist_rebut}

Domain Scientist:
{domain_scientist_rebut}

=== Closing Statements ===
Politician:
{politician_close}

Scientist:
{scientist_close}

Journalist:
{journalist_close}

Domain Scientist:
{domain_scientist_close}

Based on all arguments and evidence, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format:  
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE
"""

# === Legacy Judge Prompt ===
def user_prompt_judge_full(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    return f"""You are a neutral judge evaluating a factual debate.

Claim: {claim}

Evidence:
{evidence}

--- Opening Statements ---
Pro Agent:
{pro_open}

Con Agent:
{con_open}

--- Rebuttals ---
Pro Agent:
{pro_rebut}

Con Agent:
{con_rebut}

--- Closing Statements ---
Pro Agent:
{pro_close}

Con Agent:
{con_close}

Based on the arguments and evidence, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format: 
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE 
"""

# === Intent Enhanced Reformulation ===
def user_prompt_reformulate_pro(claim, intent):
    return f"""You support the following claim and want to restate it in a way that clearly reflects your agreement with its message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to rephrase the claim using a positive, supporting tone that affirms the claim's main idea.

Only output the reformulated claim directly, without extra explanations.

Reformulated (Pro) Claim:"""

def user_prompt_reformulate_con(claim, intent):
    return f"""You oppose the following claim and want to restate it in a way that clearly reflects your disagreement with its message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to rephrase the claim using a critical, opposing tone that challenges or rejects its main idea.

Only output the reformulated claim directly, without extra explanations.

Reformulated (Con) Claim:"""
