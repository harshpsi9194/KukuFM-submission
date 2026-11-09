"""
MANUAL FIXTURE — NOT LLM OUTPUT. For testing only. Do not use as evidence of detector behavior.

episode_001_manual_signals.py — Hand-written fixture of Episode 1 signals.
"""

import json
import os
try:
    from .schema import HookSignal, CliffhangerSignal
    from ..parser.schema import Episode, Beat
except (ImportError, ValueError):
    from candidates.schema import HookSignal, CliffhangerSignal
    from parser.schema import Episode, Beat


def get_episode_001_hook_signals() -> dict[str, HookSignal]:
    """
    Hook signals evaluated for all 45 beats of Episode 1 according to prompts/hook_detector_v1.txt.
    Dialogue hook and Visual hook (scroll-stop potential) scored independently (0–5).
    """
    scores = {
        "B001": (0, "No dialogue present.", 4, "High scroll-stop: cathedral doors part revealing bride in white silk framed in light.", True),
        "B002": (0, "No dialogue present.", 2, "Standard high wide tracking shot down the cathedral nave.", False),
        "B003": (0, "No dialogue present.", 2, "Close-up on warm smile faltering into slight confusion.", False),
        "B004": (0, "No dialogue present.", 3, "Medium profile reveals groom holding another woman's hand at altar.", True),
        "B005": (3, "Matriarch warning: 'You should turn around, Juliana.'", 2, "Matriarch in black velvet stepping forward into frame.", True),
        "B006": (1, "Single monosyllabic question: 'Why?' lacks standalone punch.", 0, "No camera or action event in this beat.", False),
        "B007": (2, "Desperate line: 'Gerald is waiting for me.'", 2, "Over-the-shoulder medium close-up.", False),
        "B008": (3, "Matriarch blunt dismissal: 'Not anymore.'", 3, "Close-up on Matriarch in ornate tiara with merciless disdain.", True),
        "B009": (2, "Pleading line: 'What do you mean?'", 2, "Close-up on stricken face.", False),
        "B010": (3, "Command: 'Look for yourself.'", 3, "Matriarch dismissive gesture; tracking shot of Juliana advancing in shock.", True),
        "B011": (0, "No dialogue present.", 2, "Panning across whispering guests in wooden pews.", False),
        "B012": (0, "No dialogue present.", 2, "Low-angle shot from behind Juliana looking up at altar couple.", False),
        "B013": (2, "Shaken whisper: 'Gerald?'", 2, "Close-up on trembling lips and tearful eyes.", False),
        "B014": (4, "Heartbroken accusation: 'Why are you holding her hand?'", 5, "Maximum visual hook: insert shot of groom's fingers interlocked with sister's gloved hand.", True),
        "B015": (5, "Explosive shocker: 'Because he is marrying me.'", 3, "Close-up on Lady in Gold Gown beaming with cruel triumph.", True),
        "B016": (2, "Gasping devastated 'No...'", 2, "Close-up as tears spill over eyelashes.", False),
        "B017": (0, "No dialogue present.", 2, "Wide profile shot below altar platform.", False),
        "B018": (4, "Outraged confrontation: 'You knew this before I walked through those doors!'", 3, "Close-up clutching chest in burning outrage.", True),
        "B019": (0, "No dialogue present.", 1, "Medium shot of Gerald standing stoic next to smug sister.", False),
        "B020": (3, "Choking on tears: 'You let me come here...'", 3, "Close-up on hands desperately clenching white dress fabric.", True),
        "B021": (4, "Grief-stricken cry: 'You let me dress in my mother's gown... believe I was walking toward my husband!'", 3, "Close-up tear-streaked face shaking with grief.", True),
        "B022": (5, "Callous rupture: 'I changed my mind.'", 4, "Tight close-up on Gerald's hardened profile looking down.", True),
        "B023": (4, "Betrayed piercing accusation: 'You swore you loved me!'", 4, "Extreme close-up on furious weeping face.", True),
        "B024": (3, "Mocking dismissal: 'People change their minds, sister.'", 2, "Close-up on Lady in Gold Gown smiling condescendingly.", True),
        "B025": (4, "Direct conspiracy confirmation: 'And you knew?' / 'Of course I knew.'", 3, "Profile shot turning head toward Gerald with satisfied malice.", True),
        "B026": (2, "Trembling question: 'How long?'", 2, "Close-up stunned by depth of conspiracy.", False),
        "B027": (5, "Cruel calculated reveal: 'Long enough to know he was never going to marry you.'", 3, "Medium two-shot of couple looking down on Juliana.", True),
        "B028": (3, "Desperate plea: 'Father...'", 3, "Action: Juliana turns abruptly to father tightly gripping brass cane.", True),
        "B029": (4, "Pleading on knees: 'Tell them this is a mistake.'", 3, "Wide shot dropping to knees beside seated father.", True),
        "B030": (0, "No dialogue present.", 2, "Close-up on father's weathered face refusing to meet her gaze.", False),
        "B031": (4, "Choking broken plea: 'Father... please look at me.'", 4, "Close-up looking up with tears streaming uncontrollably.", True),
        "B032": (0, "No dialogue present.", 2, "Father tightens jaw in stony silence.", False),
        "B033": (4, "Matriarch roaring command: 'Enough! You have embarrassed this family long enough.'", 3, "Low-angle shot looking up at commanding Matriarch.", True),
        "B034": (2, "Incredulous retort: 'Embarrassed you?'", 2, "Close-up rising from floor.", False),
        "B035": (4, "Favoritism justification: 'Clarissa knows how to carry herself... beauty, connections, future worth protecting.'", 3, "Matriarch gesturing toward Clarissa in gold.", True),
        "B036": (3, "Raw demand: 'And I am what?'", 2, "Close-up demanding answers despite humiliation.", True),
        "B037": (5, "Vicious identity blow: 'You are the daughter people tolerate because they have to. I am the daughter they choose.'", 4, "Tracking shot stepping past; close-up absorbing venomous words.", True),
        "B038": (3, "Firm declaration: 'Clarissa is the jewel of this family.'", 2, "Medium shot smiling warmly beside candles.", True),
        "B039": (4, "Dignified challenge: 'So that is what I was worth to you?'", 2, "Medium shot looking directly up at towering Gerald.", True),
        "B040": (5, "Scornful devaluation: 'You have nothing to offer me anymore.'", 4, "Close-up on Gerald's sharp jawline without remorse.", True),
        "B041": (0, "No dialogue present.", 4, "Extreme close-up on eye as single tear rolls down cheek.", True),
        "B042": (4, "Cruel insult: 'I cannot marry a woman of such plainness and low value...'", 3, "Full shot tracking Gerald turning his back on Juliana.", True),
        "B043": (5, "Arrogant dismissal: '...when Clarissa is the true jewel of the county.'", 3, "Close-up on Clarissa's radiant conceited smile.", True),
        "B044": (0, "No dialogue present.", 4, "Tight insert on white-knuckled hand clutching silk gown with suppressed fury.", True),
        "B045": (0, "No dialogue present.", 5, "Push-in close-up on tear-filled eyes hardening to cold defiance, strings swelling, cut to black.", True),
    }

    result = {}
    for b_id, (d_score, d_rat, v_score, v_rat, is_strong) in scores.items():
        result[b_id] = HookSignal(
            beat_id=b_id,
            dialogue_hook_score=d_score,
            dialogue_hook_rationale=d_rat,
            visual_hook_score=v_score,
            visual_hook_rationale=v_rat,
            is_strong_hook=is_strong
        )
    return result


def get_episode_001_cliffhanger_signals() -> dict[str, CliffhangerSignal]:
    """
    Cliffhanger signals evaluated for all 45 beats of Episode 1 according to prompts/cliffhanger_detector_v1.txt.
    Evaluates curiosity gap, specific question, over-reveal check, and taxonomy type.
    """
    scores = {
        "B001": (1, "Will Juliana walk down the aisle?", "Low", False, "none", False),
        "B002": (1, "Will Juliana reach the altar?", "Low", False, "none", False),
        "B003": (2, "What is wrong at the altar?", "Low", False, "none", False),
        "B004": (3, "Why is Gerald holding another woman's hand?", "Medium", False, "betrayal", True),
        "B005": (3, "Why is the Matriarch commanding Juliana to turn around?", "Medium", False, "threat", True),
        "B006": (1, "Why does she ask why?", "Low", False, "none", False),
        "B007": (2, "Will Juliana realize Gerald isn't waiting?", "Low", False, "none", False),
        "B008": (3, "What does 'Not anymore' mean for Juliana's marriage?", "Medium", False, "betrayal", True),
        "B009": (2, "What does the Matriarch mean?", "Low", False, "none", False),
        "B010": (3, "What shocking sight awaits Juliana as she approaches the altar?", "Medium", False, "betrayal", True),
        "B011": (1, "Why are wedding guests whispering?", "Low", False, "none", False),
        "B012": (2, "Will Juliana confront the couple?", "Low", False, "none", False),
        "B013": (2, "Will Gerald answer her whisper?", "Low", False, "none", False),
        "B014": (4, "How will Gerald explain holding another woman's hand at his wedding?", "High", False, "betrayal", True),
        "B015": (5, "How will Juliana react to Clarissa announcing she is marrying Gerald?", "High", False, "betrayal", True),
        "B016": (3, "Will Juliana collapse or fight back after the shock announcement?", "Medium", False, "betrayal", True),
        "B017": (2, "Will someone speak up from the crowd?", "Low", False, "none", False),
        "B018": (4, "Will Gerald admit he orchestrated the ambush before Juliana entered?", "High", False, "betrayal", True),
        "B019": (2, "Will Gerald break his stoic silence?", "Low", False, "none", False),
        "B020": (3, "How will Juliana hold herself together after being lured in her mother's dress?", "Medium", False, "betrayal", True),
        "B021": (4, "What callous excuse will Gerald offer after letting her believe he was her husband?", "High", False, "betrayal", True),
        "B022": (5, "How can Gerald justify throwing her away with a cold 'I changed my mind'?", "High", False, "betrayal", True),
        "B023": (4, "Will Gerald show remorse for breaking his sacred oath of love?", "High", False, "betrayal", True),
        "B024": (3, "Will Juliana strike Clarissa for mocking her heartbreak?", "Medium", False, "betrayal", True),
        "B025": (4, "How long has Clarissa been secretly plotting this theft?", "High", False, "hidden_secret", True),
        "B026": (2, "How long has the plot existed?", "Low", False, "none", False),
        "B027": (5, "Will Juliana's family defend her against this premeditated betrayal?", "High", False, "betrayal", True),
        "B028": (3, "Will her father intervene to save his daughter's honor?", "Medium", False, "moral_dilemma", True),
        "B029": (4, "Will her father speak up and declare this wedding a mistake?", "High", False, "moral_dilemma", True),
        "B030": (3, "Why is her father looking at the floor instead of his pleading daughter?", "Medium", False, "betrayal", True),
        "B031": (4, "Will her father look at her or abandon her to public shame?", "High", False, "betrayal", True),
        "B032": (4, "With her father refusing to speak, who will stand for Juliana?", "High", False, "betrayal", True),
        "B033": (4, "Will Juliana obey the Matriarch and leave in disgrace?", "High", False, "threat", True),
        "B034": (2, "How will Juliana counter the embarrassment accusation?", "Low", False, "none", False),
        "B035": (4, "Will Juliana expose Clarissa's false virtue before the Matriarch?", "High", False, "betrayal", True),
        "B036": (3, "What cruel label will the family slap on Juliana?", "Medium", False, "threat", True),
        "B037": (5, "Will Juliana accept being branded the daughter merely 'tolerated'?", "High", False, "betrayal", True),
        "B038": (3, "Will anyone dispute Clarissa being crowned the family jewel?", "Medium", False, "betrayal", True),
        "B039": (4, "What final insult will Gerald level at her worth?", "High", False, "betrayal", True),
        "B040": (5, "How will Juliana survive being told she has 'nothing to offer'?", "High", False, "betrayal", True),
        "B041": (3, "Will Juliana's tears turn into an explosive retaliation?", "Medium", False, "betrayal", True),
        "B042": (5, "Can Juliana ever recover from being publicly branded of 'low value'?", "High", False, "betrayal", True),
        "B043": (5, "Will Juliana strike back as Clarissa basks in Gerald's adoration?", "High", False, "betrayal", True),
        "B044": (4, "What will Juliana's clenched, white-knuckled fist do next?", "High", False, "imminent_action", True),
        "B045": (5, "How will Juliana execute her cold, resolute vengeance after this ultimate humiliation?", "High", False, "revenge", True),
    }

    result = {}
    for b_id, (cg_score, spec_q, stakes, defuses, c_type, is_strong) in scores.items():
        result[b_id] = CliffhangerSignal(
            beat_id=b_id,
            curiosity_gap_score=cg_score,
            specific_question=spec_q,
            stakes_importance=stakes,
            defuses_tension=defuses,
            cliffhanger_type=c_type,
            is_strong_cliffhanger=is_strong
        )
    return result
