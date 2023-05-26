from datetime import datetime

# HNU

# run1
start_datetime = '20' + '221201-00:38:15'
subjects_datetime = ['2022-12-01-01:23:19']

# run2
start_datetime = '20' + '221201-09:40:50'
subjects_datetime = ['2022-12-01-10:31:44', '2022-12-01-10:31:54']

# run3
start_datetime = '20' + '221201-11:01:32'
subjects_datetime = ['2022-12-01-11:55:57', '2022-12-01-11:56:37', '2022-12-01-12:01:58']


# run4
start_datetime = '20' + '221201-13:29:34'
subjects_datetime = ['2022-12-01-14:26:39', '2022-12-01-14:28:29', '2022-12-01-14:30:00', '2022-12-01-14:34:50']


# run5
start_datetime = '20' + '221201-15:09:33'
subjects_datetime = ['2022-12-01-16:07:09', '2022-12-01-16:11:30', '2022-12-01-16:14:00', '2022-12-01-16:15:50', '2022-12-01-16:19:21']

# run6
start_datetime = '20' + '221201-16:22:05'
subjects_datetime = ['2022-12-01-17:23:43', '2022-12-01-17:28:33', '2022-12-01-17:30:13', '2022-12-01-17:32:14', '2022-12-01-17:32:24', '2022-12-01-17:36:44']

# run7
start_datetime = '20' + '221201-22:36:09'
subjects_datetime = ['2022-12-01-23:41:27', '2022-12-01-23:49:08', '2022-12-01-23:50:58', '2022-12-01-23:54:49', '2022-12-01-23:54:49', '2022-12-01-23:59:29', '2022-12-02-00:00:19']

# run8
start_datetime = '20' + '221201-20:36:49'
subjects_datetime = ['2022-12-01-21:47:09', '2022-12-01-21:53:30', '2022-12-01-21:54:30', '2022-12-01-21:54:40', '2022-12-01-21:58:21', '2022-12-01-21:58:21', '2022-12-01-22:03:11', '2022-12-01-22:03:21']

# run9
start_datetime = '20' + '221201-06:46:19'
subjects_datetime = ['2022-12-01-07:52:09', '2022-12-01-08:02:31', '2022-12-01-08:03:01', '2022-12-01-08:05:01', '2022-12-01-08:07:41', '2022-12-01-08:09:41', '2022-12-01-08:10:31', '2022-12-01-08:10:41', '2022-12-01-08:13:42']

# run10
start_datetime = '20' + '221201-01:38:37'
subjects_datetime = ['2022-12-01-03:01:50', '2022-12-01-03:07:51', '2022-12-01-03:09:21', '2022-12-01-03:10:21', '2022-12-01-03:12:42', '2022-12-01-03:14:42', '2022-12-01-03:14:52', '2022-12-01-03:16:12', '2022-12-01-03:19:52', '2022-12-01-03:20:22']


# MSC

# run1
start_datetime = '20' + '221202-00:45:54'
subjects_datetime = ['2022-12-02-02:26:50']


# HNU_1

# run1
start_datetime = '20' + '221202-13:26:11'
dt_start = datetime.strptime(start_datetime, '%Y%m%d-%H:%M:%S')
subjects_datetime = ['2022-12-02-14:45:47']


dt_start = datetime.strptime(start_datetime, '%Y%m%d-%H:%M:%S')

run_hours = []
for success_datetime in subjects_datetime:
    # success_datetime = '2022-10-12-20:01:59'
    dt_success = datetime.strptime(success_datetime, '%Y-%m-%d-%H:%M:%S')
    dt_dif = dt_success - dt_start
    # if dt_dif.days > 0:  # 目前只统计一天的
    #     break
    run_hours.append(dt_dif.total_seconds() / 3600)

run_minutes = [i * 60 for i in run_hours]
print(run_hours)
print(run_minutes)

# import seaborn as sns
#
# sns.set_theme(style="darkgrid")
#
# # Load an example dataset with long-form data
#
# y = []
# count = len(run_hours)
# print(count)
# for i in range(1, count + 1):
#     y.append(i)
#
# # Plot the responses for different events and regions
# sns.lineplot(x=run_hours, y=y)
#
# import matplotlib.pyplot as plt
#
# plt.show()
