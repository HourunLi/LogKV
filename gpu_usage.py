# 使用方法：
# 通过资源池页面进入需要统计的卡池，点击任务列表，在云上训练作业下筛选需要统计的应用，
# 在页面下方调整每页显示条数使得所有作业在一页中显示，
# 复制所有作业（从第一条的队列位置到最后一条的云上创建时间），粘贴至gpu_usage.txt，
# 最后执行本脚本：python gpu_usage.py
with open('gpu_usage.txt', 'r', encoding='utf-8') as f:
    content = f.readlines()

user_id = None
usage = None
usage_dict = {}
user_id_dict = {
    'w50050543': '吴洲同 50050543',
    'kwx1459221': '孔博傲 wx1459221',
    'l00934246': '龙滕 00934246',
    's00945890': '孙毓森 00945890',
    'z00464769': '章伟 00464769',
    'l00638839': '李天舒 00638839',
    's50041771': '孙泽旭 50041771',
    'p00913830': '潘宇 00913830',
    'l00432262': '李祖明 00432262',
    'z00464677': '张檬 00464677',
    'z50044229': '朱俊达 50044229',
    'l00910738': '刘博晓 00910738',
    'w84315891': 'Wang Yiding 84315891',
    'l84379647': 'Lin Chengdong 84379647',
    's00465741': '宋凯凯 00465741',
    'y00586093': '姚婷 00586093',
    'q00832857': '钱诚 00832857',
    'y30067757': '于辉 30067757',
    'l00818783': '兰林 00818783',
    'g00468674': '古强 00468674',
    'w84297428': 'Wu Yimeng 84297428',
    'r84332196': 'Ran Jie 84332196',
    'l00836797': '梁渊植 00836797',
    'j00449917': '金雪松 00449917',
    'l60045770': '刘樊 60045770',
    's00435292': '孙嘉城 00435292',
    'h00584192': '黄文勇 00584192',
    'c00481843': '蔡涛 00481843',
    'y00911979': '袁淘 00911979',
    'y60037969': '杨丰华 60037969',
    'r00848717': '阮荣钜 00848717',
    'l30073293': '林洪义 30073293',
    'g30031195': '高毅 30031195',
    'b00594865': '白哲源 00594865',
    'l00523951': '刘治成 00523951',
    'c00934421': '蔡泽永 00934421',
    'm84400791': 'Mingze Li 84400791',
    'swx1389246': '商秋林',
    'l50057125': '梁钜豪',
    'cwx1481164': '陈文锴',
    'z50057415': '张鑫',
    'lwx1454865': '李厚润',
    'wwx1454864': '吴伯涵',
    'w50058739': '吴洲同'

}

i = 0
while i < len(content):
    usage = int(content[i + 8])
    user_id = content[i + 10].strip()
    user_id = user_id_dict.get(user_id, user_id)
    if user_id not in usage_dict:
        usage_dict[user_id] = 0
    usage_dict[user_id] += usage
    i += 14

usage_all = sum(usage_dict.values())
sorted_usage_list = sorted(usage_dict.items(), key=lambda x: x[1])

# group_dict = {
#     '长序列': ['冯守渤 00812354', '刘博晓 00910738', '李祖明 00432262', '于辉 30067757'],
# }
# for k, v in group_dict.items():
#     group_usage = 0
#     for user_id in v:
#         if user_id in usage_dict:
#             group_usage += usage_dict[user_id]
#             print('{:>4d}'.format(usage_dict[user_id]), user_id)
#             del usage_dict[user_id]
#     print('{:>4d}'.format(group_usage), k)
#     print('*'*25)

group_usage = 0
for k, v in sorted_usage_list:
    group_usage += v
    print('{:>4d}'.format(v), k)
# print('{:>4d}'.format(group_usage), 'others')
print('*'*25)
print('{:>4d}'.format(usage_all), 'all')

