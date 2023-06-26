def evaluate(gold_src, gold_tgt, out_complex, out_simpl, without_identical=False):
	correct, n_out, n_gold = get_num_correct_aligns(gold_src, gold_tgt,
													out_complex, out_simpl, without_identical=without_identical)

	precision = correct / n_out
	recall = correct / n_gold

	f1 = (2 * precision * recall) / (precision + recall)

	return precision, recall, n_gold, n_out, correct, f1


def get_num_correct_aligns(gold_src, gold_tgt, out_complex, out_simpl, without_identical=False):
	with open(gold_src, 'r') as gold_src_file:
		gold_complex_sents = gold_src_file.readlines()

	with open(gold_tgt, 'r') as gold_tgt_file:
		gold_simpl_sents = gold_tgt_file.readlines()

	gold_complex_sents_cln = []
	for lin in gold_complex_sents:
		if '.eoa' in lin:
			pass
		else:
			gold_complex_sents_cln.append(lin.replace('.eoa', '').strip())

	gold_simpl_sents_cln = []
	for lin in gold_simpl_sents:
		if '.eoa' in lin:
			pass
		else:
			gold_simpl_sents_cln.append(lin.replace('.eoa', '').strip())

	if len(gold_simpl_sents_cln) != len(gold_complex_sents_cln):
		raise ValueError("Wrong input, gold files have different length of content.")

	with open(out_complex, 'r') as out_complex_file:
		out_complex_sents = out_complex_file.readlines()

	with open(out_simpl, 'r') as out_simpl_file:
		out_simpl_sents = out_simpl_file.readlines()

	out_complex_sents_cln = []
	for lin in out_complex_sents:
		if '.eoa' in lin:
			pass
		else:
			out_complex_sents_cln.append(lin.replace('.eoa', '').strip())

	out_simpl_sents_cln = []
	for lin in out_simpl_sents:
		if '.eoa' in lin:
			pass
		else:
			out_simpl_sents_cln.append(lin.replace('.eoa', '').strip())

	if len(out_complex_sents_cln) != len(out_simpl_sents_cln):
		raise ValueError("Wrong input, output files have different length of content. Complex: "+str(len(out_complex_sents_cln))+", Simple: "+str(len(out_simpl_sents_cln)))

	golds = list(zip(gold_complex_sents_cln, gold_simpl_sents_cln))
	correct = 0
	aligned = len(out_complex_sents_cln)
	gold_aligned = len(gold_complex_sents_cln)
	if without_identical:
		for gold_complex, gold_simple in zip(gold_complex_sents_cln, gold_simpl_sents_cln):
			if gold_complex == gold_simple:
				gold_aligned -= 1

	for out_cmplx, out_simpl in zip(out_complex_sents_cln, out_simpl_sents_cln):
		if without_identical and out_cmplx == out_simpl:
			aligned -= 1
			continue
		if (out_cmplx, out_simpl) in golds:
			correct += 1

	return correct, aligned, gold_aligned


def evaluate_n_m(gold_src, gold_tgt, out_complex, out_simpl, types, without_identical=False):
	correct, n_out, n_gold, n_11, n_1m, n_n1, correct_n_11, correct_n_1m, correct_n_n1 = get_num_correct_aligns_n_m(
        gold_src, 
        gold_tgt,
        out_complex, 
        out_simpl,
        types,
        without_identical=without_identical
    )

	precision = correct / n_out
	recall = correct / n_gold

	f1 = (2 * precision * recall) / (precision + recall)

	return precision, recall, n_gold, n_out, correct, f1, n_11, n_1m, n_n1, correct_n_11, correct_n_1m, correct_n_n1


def get_num_correct_aligns_n_m(gold_src, gold_tgt, out_complex, out_simpl, types, without_identical=False):
    with open(gold_src, 'r') as gold_src_file:
        gold_complex_sents = gold_src_file.readlines()
    
    with open(gold_tgt, 'r') as gold_tgt_file:
        gold_simpl_sents = gold_tgt_file.readlines()
    
    gold_complex_sents_cln = []
    for lin in gold_complex_sents:
        if '.eoa' in lin:
            pass
        else:
            gold_complex_sents_cln.append(lin.replace('.eoa', '').strip())
    
    gold_simpl_sents_cln = []
    for lin in gold_simpl_sents:
        if '.eoa' in lin:
            pass
        else:
            gold_simpl_sents_cln.append(lin.replace('.eoa', '').strip())
    
    if len(gold_simpl_sents_cln) != len(gold_complex_sents_cln):
        raise ValueError("Wrong input, gold files have different length of content.")
    
    with open(out_complex, 'r') as out_complex_file:
        out_complex_sents = out_complex_file.readlines()
    
    with open(out_simpl, 'r') as out_simpl_file:
        out_simpl_sents = out_simpl_file.readlines()
    
    with open(types, 'r') as out_typ_file:
        out_typ_vals = out_typ_file.readlines()
    
    out_complex_sents_cln = []
    for lin in out_complex_sents:
        if '.eoa' in lin:
            pass
        else:
            out_complex_sents_cln.append(lin.replace('.eoa', '').strip())

    out_simpl_sents_cln = []
    for lin in out_simpl_sents:
        if '.eoa' in lin:
            pass
        else:
            out_simpl_sents_cln.append(lin.replace('.eoa', '').strip())

    if len(out_complex_sents_cln) != len(out_simpl_sents_cln) != len(out_typ_vals):
        raise ValueError("Wrong input, output files have different length of content. Complex: "+str(len(out_complex_sents_cln))+", Simple: "+str(len(out_simpl_sents_cln))+", Types: "+str(len(out_typ_vals)))

    golds = list(zip(gold_complex_sents_cln, gold_simpl_sents_cln))
    correct = 0
    aligned = len(out_complex_sents_cln)
    gold_aligned = len(gold_complex_sents_cln)
    if without_identical:
        for gold_complex, gold_simple in zip(gold_complex_sents_cln, gold_simpl_sents_cln):
            if gold_complex == gold_simple:
                gold_aligned -= 1

    n_11 = n_1m = n_n1 = correct_n_11 = correct_n_1m = correct_n_n1 = 0
    for out_cmplx, out_simpl, ty in zip(out_complex_sents_cln, out_simpl_sents_cln, out_typ_vals):
        if ty.strip() == '1:1':
            n_11 += 1
        elif ty.strip() == '1:m':
            n_1m += 1
        elif ty.strip() == 'n:1':
            n_n1 += 1
        if without_identical and out_cmplx == out_simpl:
            aligned -= 1
            continue
        if (out_cmplx, out_simpl) in golds:
            correct += 1
            if ty.strip() == '1:1':
                correct_n_11 += 1
            elif ty.strip() == '1:m':
                correct_n_1m += 1
            elif ty.strip() == 'n:1':
                correct_n_n1 += 1

    return correct, aligned, gold_aligned, n_11, n_1m, n_n1, correct_n_11, correct_n_1m, correct_n_n1
