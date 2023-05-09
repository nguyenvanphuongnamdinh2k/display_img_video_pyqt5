import csv

# Tên file CSV
filename = "phuong.csv"
a = {'A':1,'B':2,'C':3}
# Mở file CSV và thiết lập mode là ghi ('w')
# with open(filename, 'w', newline='') as csvfile:
#
#     # Thiết lập đối tượng writer
#     csvwriter = csv.writer(csvfile)
#
#     # Viết tiêu đề của các cột
#     csvwriter.writerow(['A', 'B'])
#
#     # Viết dữ liệu cho mỗi dòng
#     csvwriter.writerow(['a1', 'b1'])
#     csvwriter.writerow(['a2', 'b2'])
with open(filename,mode="a",newline='') as file:
    writer = csv.writer(file)
    writer.writerow(a.keys())
    writer.writerow([a.get('A', ''), a.get('B', '')])