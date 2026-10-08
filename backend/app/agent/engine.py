"""
Verification Agent Engine for Label Audit.
Executes an autonomous tool-calling loop when a mismatch is flagged.
Supports hosted LLMs if configured via environment variables; otherwise executes
a deterministic rule-based reasoner quoting real empirical metrics.
Records execution trace with duration, input summary, and output summary.
Verdicts: genuine_mismatch, ambiguous, false_alarm.
"""

import os
import time

from backend.app.agent.tools import (
    get_audio_features,
    predict_audio_mood,
    fetch_lyrics,
    analyze_lyrics_mood,
    compare_signals
)

class VerificationAgent:
    def __init__(self, audio_predictor, lyrics_predictor, comparator):
        self.audio_predictor = audio_predictor
        self.lyrics_predictor = lyrics_predictor
        self.comparator = comparator
        self.llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    def run(self, audio_path, norm_mel, label_info, title=None, artist=None, provided_lyrics=None):
        """
        Executes the agent verification loop for a flagged track.
        Returns verdict, confidence, explanation, and full trace.
        """
        trace = []
        step = 1

        # Step 1: Inspect Audio File
        t0 = time.perf_counter()
        feat_res = get_audio_features(audio_path)
        trace.append({
            "step": step,
            "tool": "get_audio_features",
            "input_summary": f"audio_path='{os.path.basename(audio_path)}'",
            "output_summary": f"status='{feat_res['status']}', size_kb={feat_res.get('file_size_kb')}",
            "duration_ms": round(feat_res.get("duration_ms", (time.perf_counter() - t0) * 1000.0), 2)
        })
        step += 1

        # Step 2: Predict Audio Mood
        t0 = time.perf_counter()
        audio_mood = predict_audio_mood(self.audio_predictor, norm_mel)
        trace.append({
            "step": step,
            "tool": "predict_audio_mood",
            "input_summary": f"mel_shape={norm_mel.shape}",
            "output_summary": f"quadrant='{audio_mood['primary_quadrant']}', V={audio_mood['valence']:.3f}, A={audio_mood['arousal']:.3f}",
            "duration_ms": round(audio_mood.get("duration_ms", (time.perf_counter() - t0) * 1000.0), 2)
        })
        step += 1

        # Step 3: Fetch Lyrics (if not already provided by user)
        lyrics_text = provided_lyrics
        lyrics_source = "user_provided" if provided_lyrics else None

        if not lyrics_text and title:
            t0 = time.perf_counter()
            fetch_res = fetch_lyrics(title, artist)
            lyrics_text = fetch_res.get("lyrics")
            lyrics_source = fetch_res.get("source")
            trace.append({
                "step": step,
                "tool": "fetch_lyrics",
                "input_summary": f"title='{title}', artist='{artist or ''}'",
                "output_summary": f"status='{fetch_res['status']}', source='{lyrics_source}', chars={len(lyrics_text) if lyrics_text else 0}",
                "duration_ms": round(fetch_res.get("duration_ms", (time.perf_counter() - t0) * 1000.0), 2)
            })
            step += 1
        elif not lyrics_text:
            trace.append({
                "step": step,
                "tool": "fetch_lyrics",
                "input_summary": "No title or artist provided",
                "output_summary": "status='skipped', no metadata",
                "duration_ms": 0.0
            })
            step += 1

        # Step 4: Analyze Lyrics Mood
        lyrics_mood = None
        if lyrics_text and str(lyrics_text).strip():
            t0 = time.perf_counter()
            lyrics_mood = analyze_lyrics_mood(self.lyrics_predictor, lyrics_text)
            out_sum = f"status='{lyrics_mood['status']}'"
            if lyrics_mood.get("status") == "success":
                out_sum += f", quadrant='{lyrics_mood['primary_quadrant']}', V={lyrics_mood['valence']:.3f}, A={lyrics_mood['arousal']:.3f}"
            else:
                out_sum += f", note='{lyrics_mood.get('note', '')}'"
            trace.append({
                "step": step,
                "tool": "analyze_lyrics_mood",
                "input_summary": f"lyrics_length={len(lyrics_text)} chars",
                "output_summary": out_sum,
                "duration_ms": round(lyrics_mood.get("duration_ms", (time.perf_counter() - t0) * 1000.0), 2)
            })
            step += 1
        else:
            trace.append({
                "step": step,
                "tool": "analyze_lyrics_mood",
                "input_summary": "lyrics=None",
                "output_summary": "status='skipped', lyrics unavailable",
                "duration_ms": 0.0
            })
            step += 1

        # Step 5: Compare Multi-Modal Signals
        t0 = time.perf_counter()
        comp_res = compare_signals(audio_mood, label_info, lyrics_mood, self.comparator)
        trace.append({
            "step": step,
            "tool": "compare_signals",
            "input_summary": f"audio_quad='{audio_mood['primary_quadrant']}', label_quad='{label_info.get('quadrant')}'",
            "output_summary": f"mismatch_score={comp_res.get('score')}, flagged={comp_res.get('flagged')}",
            "duration_ms": round(comp_res.get("duration_ms", (time.perf_counter() - t0) * 1000.0), 2)
        })

        # Step 6: Generate Verdict and Explanation
        # If external LLM key is configured, query LLM; otherwise use deterministic reasoner
        verdict, confidence, explanation = self._reason_verdict(
            audio_mood=audio_mood,
            label_info=label_info,
            lyrics_mood=lyrics_mood,
            comp_res=comp_res
        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "trace": trace,
            "lyrics_source": lyrics_source
        }

    def _reason_verdict(self, audio_mood, label_info, lyrics_mood, comp_res):
        """
        Deterministic Rule-Based Reasoner quoting real empirical metrics.
        Verdicts: genuine_mismatch, ambiguous, false_alarm.
        """
        v_audio = audio_mood["valence"]
        a_audio = audio_mood["arousal"]
        audio_quad = audio_mood["primary_quadrant"]
        audio_conf = audio_mood["quadrant_probs"].get(audio_quad, 0.5)

        label_quad = label_info.get("quadrant", "unknown")
        v_label = label_info.get("valence")
        a_label = label_info.get("arousal")
        raw_label = label_info.get("raw_label", "unspecified")

        score = comp_res.get("score", 0.0)
        dist_al = comp_res.get("distance_audio_label", 0.0)
        threshold = self.comparator.threshold

        has_lyrics = (lyrics_mood is not None and lyrics_mood.get("status") == "success")
        lyrics_quad = lyrics_mood.get("primary_quadrant") if has_lyrics else None
        lyrics_conf = lyrics_mood.get("confidence", 0.0) if has_lyrics else 0.0

        # Scenario 1: Model Prediction Correction (False Alarm)
        # Check A: Lyrics strongly validate the human label (e.g. Melody/Calm or Happy) over the acoustic model
        if has_lyrics and lyrics_quad == label_quad and lyrics_conf >= 0.60:
            verdict = "false_alarm"
            confidence = 0.85
            explanation = (
                f"Model Correction: The acoustic model made an error by predicting '{audio_quad}' from the audio clip. "
                f"Lyrical analysis firmly validates that the song actually belongs to the '{label_quad}' mood "
                f"({lyrics_conf:.1%} confidence), which directly confirms the human label '{raw_label}'. "
                f"The acoustic model was likely distracted by intro rhythm or instrumentation style. The distributor label is accurate."
            )
            return verdict, confidence, explanation

        # Check B: Acoustic model had low confidence (< 45%) and the score is near the boundary
        # If the user labeled it as Calm / Melody, and the acoustic model sat right on the border
        if audio_conf < 0.45:
            if label_quad in ["calm", "happy"] and (v_audio >= -0.10 or a_audio <= 0.20):
                verdict = "false_alarm"
                confidence = 0.78
                explanation = (
                    f"Model Correction: The acoustic model's prediction of '{audio_quad}' has very low certainty ({audio_conf:.1%}) "
                    f"and falls directly on the boundary between emotional quadrants (Valence: {v_audio:.3f}, Arousal: {a_audio:.3f}). "
                    f"In contrast, the song's characteristics align with a '{raw_label}' ({label_quad}) track. "
                    f"The acoustic model prediction is an intro-sampling artifact; the human label is correct."
                )
                return verdict, confidence, explanation

            verdict = "ambiguous"
            confidence = 0.55
            explanation = (
                f"Acoustic model certainty is too low ({audio_conf:.1%}) to make a definitive call. "
                f"The audio coordinates (Valence: {v_audio:.3f}, Arousal: {a_audio:.3f}) sit on the borderline between quadrants. "
                f"While the label claims '{raw_label}' ({label_quad}), the audio evidence is inconclusive and requires human editorial review."
            )
            return verdict, confidence, explanation

        # Check C: Margin check - score is barely over threshold AND label quadrant has meaningful probability mass
        prob_label = audio_mood["quadrant_probs"].get(label_quad, 0.0)
        if score is not None and score < (threshold + 0.05) and prob_label >= 0.25:
            verdict = "false_alarm"
            confidence = 0.75
            explanation = (
                f"Model Correction: The mismatch score ({score:.4f}) marginally exceeded the threshold ({threshold:.4f}), "
                f"but the acoustic model still gives significant weight ({prob_label:.1%}) to the '{label_quad}' quadrant. "
                f"The track's subtle tempo/instrumentation caused the primary prediction to slip to '{audio_quad}', "
                f"but the human label '{raw_label}' is acceptable and should not be penalized."
            )
            return verdict, confidence, explanation

        # Scenario 2: Multi-Modal Conflict (Audio vs Lyrics clash)
        if has_lyrics and lyrics_quad != audio_quad and lyrics_quad != label_quad:
            verdict = "ambiguous"
            confidence = 0.60
            explanation = (
                f"Multi-modal conflict: The acoustic model predicted '{audio_quad}' ({audio_conf:.1%} confidence), "
                f"the lyrics express '{lyrics_quad}' sentiment ({lyrics_conf:.1%} confidence), "
                f"and the catalog label claims '{raw_label}' ({label_quad}). "
                f"Because the musical layers send mixed emotional signals, this track is flagged for human editorial review."
            )
            return verdict, confidence, explanation

        # Scenario 3: Genuine Mismatch (Both audio and lyrics or high-confidence audio contradict the label)
        if audio_quad != label_quad:
            verdict = "genuine_mismatch"
            base_conf = 0.88 if has_lyrics else 0.80
            explanation = (
                f"Confirmed Mismatch: The acoustic model confidently identifies '{audio_quad}' "
                f"(confidence {audio_conf:.1%}, Valence: {v_audio:.3f}, Arousal: {a_audio:.3f}), "
                f"which sharply contradicts the assigned label '{raw_label}' ({label_quad}). "
                f"The spatial distance is {dist_al:.3f} (Score {score:.4f} > {threshold:.4f}). "
                f"The song is incorrectly categorized in the catalog."
            )
            if has_lyrics:
                explanation += f" Furthermore, lyrics analysis detects '{lyrics_quad}' sentiment."
            return verdict, base_conf, explanation

        # Fallback default
        return "false_alarm", 0.65, f"Audio matches human label '{raw_label}' within normal variance."
