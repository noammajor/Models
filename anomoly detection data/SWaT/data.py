import xlrd
import csv

'''def xlsx_to_csv():
    workbook = xlrd.open_workbook('SWaT_Dataset_Normal_v1.xlsx')
    table = workbook.sheet_by_index(0)
    with open('swat_train.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)

if __name__ == '__main__':
    xlsx_to_csv()'''
'''import pandas as pd
import csv
f = open('swat_train2.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(['FIT101' ,'LIT101' ,' MV101', 'P101' ,'P102', ' AIT201', 'AIT202',
 'AIT203' ,'FIT201' ,' MV201' ,' P201', ' P202', 'P203' ,' P204', 'P205', 'P206',
 'DPIT301' ,'FIT301', 'LIT301' ,'MV301' ,'MV302', ' MV303', 'MV304' ,'P301'
, 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
 'UV401' ,'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502' ,'FIT503',
 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602',
 'P603', 'Normal/Attack'])

df = pd.read_csv('swat_train.csv')
df = df.values
for u in range(1,len(df)):
    tem = df[u][1:]
    tem2 = []
    for i in range(len(tem)-1):
        tem2.append(float(tem[i]))
    if(tem[-1]=='Normal'):
        tem2.append(0)
    else:
        tem2.append(1)
    csv_writer.writerow(tem2)'''
import pandas as pd
df = pd.read_csv('swat_train2.csv')
df = df.values
print(df.shape)
ano = 0
for u in range(len(df)):
    if(df[u][-1]==1):
        ano = ano+1
print(ano)
print(ano/len(df))


