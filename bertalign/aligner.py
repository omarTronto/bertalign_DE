import numpy as np
from tqdm import tqdm
from bertalign import model
from bertalign.corelib import *
from bertalign.utils import *

class Bertalign:
    def __init__(self,
                 src,
                 tgt,
                 max_align=5,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
               ):
        
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        
        #src = clean_text(src)
        #tgt = clean_text(tgt)
        
        src_lang='de'
        tgt_lang='de'
        
        #if is_split:
        #    src_sents = src.splitlines()
        #    tgt_sents = tgt.splitlines()
        #else:
        #    src_sents = split_sents(src, src_lang)
        #    tgt_sents = split_sents(tgt, tgt_lang)
        
        src_sents = src
        tgt_sents = tgt
 
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]
        
        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        print("Embedding source and target text using {} ...".format(model.model_name))
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
        
    def align_sents(self):

        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        print("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line


def get_n_m_alignments(
    complex_file, 
    simple_file,
    output_file,
    allow_1_1 = True,
    allow_1_m = True,
    allow_n_1 = True
):
    def get_alignments(src_doc_lines, tgt_doc_lines):
        aligner = Bertalign(src_doc_lines, tgt_doc_lines)
        aligner.align_sents()
        results = aligner.result
        alignments = []
        compl_list = []
        simpl_list = []
        for entry in results:
            for c_sent_ind in entry[0]:
                for s_sent_ind in entry[1]:
                    compl_list.append(src_doc_lines[c_sent_ind])
                    simpl_list.append(tgt_doc_lines[s_sent_ind])
                    alignments.append((src_doc_lines[c_sent_ind], tgt_doc_lines[s_sent_ind]))
        return alignments, compl_list, simpl_list
    
    with open(complex_file, 'r') as srcf:
        with open(simple_file, 'r') as tgtf:
            src_lines = srcf.readlines()
            tgt_lines = tgtf.readlines()

            src_doc_lines = []
            src_all_docs_lines = []
            for src_line in src_lines:
                if '.eoa' in src_line:
                    src_all_docs_lines.append(src_doc_lines)
                    src_doc_lines = []
                else:
                    src_doc_lines.append(src_line.strip())

            tgt_doc_lines = []
            tgt_all_docs_lines = []
            for tgt_line in tgt_lines:
                if '.eoa' in tgt_line:
                    tgt_all_docs_lines.append(tgt_doc_lines)
                    tgt_doc_lines = []
                else:
                    tgt_doc_lines.append(tgt_line.strip())

            assert len(src_all_docs_lines) == len(tgt_all_docs_lines), "Number of docs in source file is not equal to the number of docs in target file"

            alignments_all = []
            n_11 = n_1m = n_n1 = 0 
            for src_doc_lines, tgt_doc_lines in tqdm(zip(src_all_docs_lines, tgt_all_docs_lines)):
                alignments, compl_list, simpl_list = get_alignments(src_doc_lines, tgt_doc_lines)
                
                alignments__ = []
                i = 0
                while i < len(alignments):
                    # get list of indicies block in complex
                    j = i + 1
                    compl_indices_list = [i]
                    while j < len(alignments):
                        if compl_list[i] == compl_list[j]:
                            compl_indices_list.append(j)
                            j = j + 1
                        else:
                            break
                    
                    # get list of indicies block in simple
                    j = i + 1
                    simpl_indices_list = [i]
                    while j < len(alignments):
                        if simpl_list[i] == simpl_list[j]:
                            simpl_indices_list.append(j)
                            j = j + 1
                        else:
                            break
                            
                    if len(simpl_indices_list) == 1 and len(compl_indices_list) > 1: # 1:m
                        if allow_1_m:
                            comp_sent__ = compl_list[compl_indices_list[0]]
                            simp_sent__ = ''
                            for ind in compl_indices_list:
                                simp_sent__ = simp_sent__ + ' ' + simpl_list[ind]
                            simp_sent__ = simp_sent__.strip()
                            alignments__.append((comp_sent__, simp_sent__, '1:m'))
                            i = i + len(compl_indices_list)
                            n_1m += 1
                        else:
                            i = i + len(compl_indices_list)
                            continue
                        
                    elif len(compl_indices_list) == 1 and len(simpl_indices_list) > 1: # n:1
                        if allow_n_1: 
                            comp_sent__ = ''
                            simp_sent__ = simpl_list[simpl_indices_list[0]]
                            for ind in simpl_indices_list:
                                comp_sent__ = comp_sent__ + ' ' + compl_list[ind]
                            comp_sent__ = comp_sent__.strip()
                            alignments__.append((comp_sent__, simp_sent__, 'n:1'))
                            i = i + len(simpl_indices_list)
                            n_n1 += 1
                        else:
                            i = i + len(simpl_indices_list)
                            continue
                        
                    elif len(compl_indices_list) == 1 and len(simpl_indices_list) == 1: # 1:1
                        if allow_1_1:
                                alignments__.append(
                                    (compl_list[compl_indices_list[0]], 
                                     simpl_list[simpl_indices_list[0]], '1:1')
                                )
                                i = i + 1
                                n_11 += 1
                        else:
                            i = i + 1
                            continue
                        
                    else:
                        raise Exception(f'Algorithm break cause len(compl_indices_list) = {len(compl_indices_list)} and len(simpl_indices_list) = {len(simpl_indices_list)}')
                
                alignments_all.extend(alignments__)
            
    with open(output_file+'.complex', 'w') as outcf:
        with open(output_file+'.simpl', 'w') as outsf:
            with open(output_file+'.types', 'w') as outty:
                for entry in alignments_all:
                    outcf.write(entry[0]+'\n')
                    outsf.write(entry[1]+'\n')
                    outty.write(entry[2]+'\n')
        
    return alignments_all, output_file+'.complex', output_file+'.simpl', n_11, n_1m, n_n1
    