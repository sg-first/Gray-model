import xlrd

data = xlrd.open_workbook('数据.xlsx')
table = data.sheet_by_name('Sheet1')

def getCell(x,y):
    val=table.cell(x, y).value
    return val

def writeFile(filePath, content):
    print('Write info to file:Start...')
    # 将文件内容写到文件中
    with open(filePath, 'a', encoding='utf-8') as f:
        f.write(content)
        print('Write info to file:end...')

def readFile(filepath):
    f=open(filepath,'r')
    return f.read()