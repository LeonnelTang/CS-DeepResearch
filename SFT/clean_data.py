import json

def remove_last_message_from_jsonl(input_file, output_file):
    """
    去除JSONL文件中每条数据的messages数组的最后一个JSON对象
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())

                # 检查是否有messages字段且长度大于1
                if 'messages' in data and len(data['messages']) > 1:
                    # 去除最后一个message
                    data['messages'] = data['messages'][:-1]

                # 写入处理后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
                continue

# 使用示例
input_file = r"D:\DissertationProject\DissertationProject\SFT\data\hotpotqa.jsonl"
output_file = r"D:\DissertationProject\DissertationProject\SFT\data\hotpotqa_cleaned.jsonl"

remove_last_message_from_jsonl(input_file, output_file)
print(f"处理完成！输出文件保存为: {output_file}")
