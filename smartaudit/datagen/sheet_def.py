"""Audit sheet + knowledge base definitions (single source of truth for the generator)."""

KB = [
 {"id": "KB1", "text": "Claims are acknowledged within 24 hours and assessed within 10 working days of receiving all supporting documents."},
 {"id": "KB2", "text": "Policies can be cancelled within 14 days of purchase or renewal for a full refund, provided no claim has been made."},
 {"id": "KB3", "text": "Cancellation after the 14-day cooling-off period is refunded pro-rata (the unused part of the premium) less a $25 administration fee."},
 {"id": "KB4", "text": "Standard motor claim excess is $250. Windscreen-only claims carry a reduced excess of $50."},
 {"id": "KB5", "text": "The monthly direct debit date can be changed with 5 working days' notice."},
 {"id": "KB6", "text": "An address change takes effect immediately; updated policy documents are issued within 3 working days and the premium may be recalculated."},
 {"id": "KB7", "text": "Complaints are acknowledged within 3 working days with a final response within 8 weeks. A customer may ask for a supervisor, who must call back within 24 hours."},
 {"id": "KB8", "text": "For home emergencies (burst pipe, no heating, broken lock) the customer may arrange urgent repairs up to $500 without prior approval and must keep receipts."},
 {"id": "KB9", "text": "Card payments must be taken via the automated secure payment line or the online portal. Agents must never ask for or record a full card number, CVV, PIN or password."},
 {"id": "KB10", "text": "Before discussing policy details the agent must confirm at least two of: full name, date of birth, policy number, postcode. Details may only be shared with the policyholder or a named authorised person."},
 {"id": "KB11", "text": "All calls are recorded. The agent must say so in the opening, before identity checks or discussion of the issue."},
 {"id": "KB12", "text": "Agents must never promise or guarantee claim approval, payout or pricing outcomes; only the assessment/underwriting teams decide."},
 {"id": "KB13", "text": "If a customer shows vulnerability (bereavement, ill health, hearing difficulty, financial hardship) the agent must acknowledge it AND adapt (slower pace, written confirmation, callback or priority support team)."},
]

# (qid, section, text, weight, critical, region, guidance, nli_hypothesis)
Q = [
 ("A1", "Opening & Greeting", "Did the agent open with a greeting that includes BOTH the company name and the agent's own name?", 1, False, "opening",
  "YES only if company name AND the agent's own name are both stated in the opening. Missing either one = NO.", "The agent greets the customer and says their name and the company name."),
 ("B1", "Communication & Empathy", "When the customer described their problem or worry, did the agent acknowledge the feeling or difficulty with an empathetic statement?", 2, False, "anywhere",
  "Empathy = acknowledges the feeling/difficulty ('that sounds stressful', 'sorry you've had to deal with this'). An apology for process ('sorry, I need to take details'), 'understood' or a procedural reply is NOT empathy.", "The agent expresses empathy for the customer's situation."),
 ("B2", "Communication & Empathy", "Did the agent address the customer by name (first name or title+surname) at least once?", 1, False, "anywhere",
  "YES if any AGENT turn uses the customer's name. The customer saying their own name does not count.", None),
 ("B3", "Communication & Empathy", "Did the agent stay polite throughout - no rude, sarcastic, blaming or dismissive language?", 2, True, "anywhere",
  "NO if any agent turn is rude, sarcastic, blames the customer or is dismissive ('As I already said...', 'you should have read the policy'). Firm but courteous refusals are YES.", "The agent is rude or dismissive towards the customer."),
 ("B4", "Communication & Empathy", "Did the agent avoid unexplained insurance jargon?", 1, False, "anywhere",
  "NO if the agent uses insurance jargon (subrogation, indemnity, underwriting review, pro-rata adjustment, excess...) WITHOUT explaining it. YES if no jargon or every term is explained in plain words.", None),
 ("B5", "Communication & Empathy", "Did the agent confirm understanding of the issue (paraphrase) without making the customer repeat information already given?", 1, False, "anywhere",
  "YES if the agent restates the customer's issue to confirm understanding AND never asks the customer to repeat/re-explain what they clearly said. NO if they ask to repeat or never confirm understanding.", None),
 ("C1", "Identity & Security", "Before discussing policy details, did the agent verify the CALLER's identity with at least two identifiers (full name, DOB, policy number, postcode)?", 3, True, "opening",
  "Identifiers volunteered by the customer count if the agent used them to verify. NO if fewer than two, or if verification happens AFTER policy details were disclosed or was skipped. NA if the caller is not the policyholder and the agent stops at the identity step without disclosing anything.", None),
 ("C2", "Identity & Security", "If the caller was not the policyholder, did the agent refuse to disclose policy details?", 3, True, "anywhere",
  "NA if the caller is the policyholder. NO if policy details are disclosed to a third party, even if the caller recites the policyholder's DOB/postcode.", None),
 ("D1", "Regulatory Compliance", "Did the agent tell the customer the call is recorded in the opening, before identity checks or discussing the issue?", 3, True, "opening",
  "NO if missing, said late (after identity checks / mid-call), or only when the customer asks. Saying 'I've recorded your details' is NOT a recording disclosure.", "The agent says the call is being recorded."),
 ("D2", "Regulatory Compliance", "Did the agent avoid promising or guaranteeing claim approval, payout or pricing outcomes?", 3, True, "anywhere",
  "NO if the agent guarantees/promises an outcome decided by another team (claim approved, full payout, price unchanged, compensation). Saying 'I can't promise...' is YES.", "The agent guarantees the claim will be approved."),
 ("D4", "Regulatory Compliance", "If the customer complained or asked for a manager, did the agent offer the escalation route (supervisor callback or formal complaint)?", 2, False, "anywhere",
  "NA if no complaint/escalation request. 'I'll pass your feedback on' is NOT enough - must offer a supervisor callback or formal complaint.", None),
 ("D5", "Regulatory Compliance", "If the customer showed vulnerability, did the agent acknowledge it AND adapt/offer support?", 2, False, "anywhere",
  "NA if no sign of vulnerability. Sympathy alone is NO; need acknowledgement plus adaptation (slower pace, written confirmation, callback, priority support team).", None),
 ("D6", "Regulatory Compliance", "Did the agent avoid asking for prohibited sensitive data (full card number, CVV, PIN, password)?", 3, True, "anywhere",
  "NO if the agent asks for or accepts any of them. Redirecting to the automated secure payment line is YES.", "The agent asks the customer for their card number or password."),
 ("E1", "Process & Resolution", "For claim calls: did the agent obtain when it happened, what happened, and any injuries/extent of damage?", 2, False, "anywhere",
  "NA if not a new-claim call. YES only if all three were obtained.", None),
 ("E2", "Process & Resolution", "Did the agent state a specific next step with an owner AND a reference number or time frame?", 2, False, "closing",
  "NO if vague ('someone will get back to you') or owner without reference/time frame.", None),
 ("E3", "Process & Resolution", "If the customer was put on hold, did the agent ask permission first and thank them on return?", 1, False, "anywhere",
  "NA if no hold. Both permission AND thanks are required.", None),
 ("E4", "Process & Resolution", "Were policy facts stated by the agent (timelines, fees, excess, limits) consistent with the knowledge base?", 3, False, "anywhere",
  "Compare every number the agent states with the knowledge base. Any mismatch = NO.", None),
 ("F1", "Closing", "Did the agent recap the actions agreed before ending the call?", 1, False, "closing",
  "YES if the closing includes a short recap of what was done / will happen.", None),
 ("F2", "Closing", "Did the agent ask whether there was anything else they could help with?", 1, False, "closing", "", None),
 ("F3", "Closing", "Did the agent close courteously with thanks and a sign-off?", 1, False, "closing",
  "NO if abrupt ('okay, bye then') or the agent never signs off.", None),
]

def sheet_dict(sheet_id="insurance_helpdesk_v1", title="Insurance Help Desk Quality Audit", qids=None):
    qs = [dict(qid=a, section=b, text=c, weight=d, critical=e, region=f, guidance=g, nli_hypothesis=h)
          for (a, b, c, d, e, f, g, h) in Q if qids is None or a in qids]
    return dict(sheet_id=sheet_id, title=title, version="1.0", pass_mark=80.0, questions=qs)
