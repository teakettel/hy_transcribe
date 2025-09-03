import jiwer
import collections

# TODO: This file is ugly as fuck
# Add a couple extra behaviors on top of jiwer
class WerAlignment(): 
    
    def __init__(self, reference_text, hypothesis_text): 
        word_output = jiwer.process_words([reference_text],[hypothesis_text]) 
        self.alignment = word_output.alignments[0]
        self.reference = word_output.references[0]
        self.hypothesis = word_output.hypotheses[0]
        self.wer = round(word_output.wer, 2)
        self.n_sub = word_output.substitutions 
        self.n_ins = word_output.insertions 
        self.n_del = word_output.deletions
        self.ref_wc = len(self.reference) 
        self.hyp_wc = len(self.hypothesis)
        self.n_corr = self.hyp_wc - (self.n_ins + self.n_sub)
        self.n_err = self.n_ins + self.n_sub + self.n_del

    def all_scores(self):
        return {
            "wer": self.wer,
            "n_errors": self.n_err,
            "n_correct": self.n_corr,
            "n_sub": self.n_sub,
            "n_ins": self.n_ins,
            "n_del": self.n_del,
            "word_count": self.ref_wc,
            "hyp_word_count": self.hyp_wc
        }
    
    def n_likely_hallucinations(self, min_sequence_length=5): 
        total = 0
        running_seq_len = 0
        for chunk in self.alignment: 
            if chunk.type in ['insert', 'substitute']: 
                running_seq_len += (chunk.hyp_end_idx - chunk.hyp_start_idx) 
            else: 
                if running_seq_len >= min_sequence_length: 
                    total += running_seq_len
                running_seq_len = 0 

        if running_seq_len >= min_sequence_length: 
            total += running_seq_len

        return total

    def print_alignment(self): 
        lines = []
        for chunk in self.alignment: 
            reference_chunk = self.reference[chunk.ref_start_idx:chunk.ref_end_idx]
            hypothesis_chunk = self.hypothesis[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if chunk.type == 'insert': 
                for word in hypothesis_chunk: 
                    lines.append(f">\t>\t{word}")
            elif chunk.type == 'delete': 
                for word in reference_chunk: 
                    lines.append(f"<\t{word}\t<")
            else:
                for i, word in enumerate(reference_chunk): 
                    hyp = hypothesis_chunk[i]
                    symbol = "=" if word == hyp else "!"
                    lines.append(f"{symbol}\t{word}\t{hyp}")
        print("\n".join(lines))