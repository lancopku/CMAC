import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file_path', default='prediction.json')
parser.add_argument('-hypo_output', default='hypo.txt')
parser.add_argument('-ref_output', default='ref.txt')
opt = parser.parse_args()

output = json.load(open(opt.file_path, 'r', encoding='utf-8'))
out_hypo = open(opt.hypo_output, 'w', encoding='utf-8')
out_ref = open(opt.ref_output, 'w', encoding='utf-8')
for data in output:
    predict = data['prediction']
    target = data['target'][6:]  # get rid of <BOS>
    out_hypo.write(predict + '\n')
    out_ref.write(target + '\n')

out_hypo.close()
out_ref.close()
