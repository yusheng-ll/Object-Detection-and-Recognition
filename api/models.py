from django.db import models

# 这张表：存储每次的检测记录（图片、人数、密度、状态、时间）
class DetectRecord(models.Model):
    id = models.BigAutoField(primary_key=True)
    filename = models.CharField(max_length=255)          # 文件名
    file_path = models.CharField(max_length=500)         # 图片路径 / 视频路径
    crowd_count = models.IntegerField()                  # 检测出的人数
    density_level = models.CharField(max_length=50)       # 密度等级 / 数值
    is_abnormal = models.BooleanField(default=False)     # 是否拥挤
    create_time = models.DateTimeField(auto_now_add=True)# 检测时间
    is_video = models.BooleanField(default=False)        # 标记是否为视频

    class Meta:
        db_table = 'detect_record'
        ordering = ['-create_time']

    def __str__(self):
        return f"第{self.crowd_count}人 | {self.create_time}"

# 训练日志表
#class TrainLog(models.Model):
#    id = models.BigAutoField(primary_key=True)
#    epoch = models.IntegerField()
#    loss = models.FloatField()
#    mae = models.FloatField()
#    create_time = models.DateTimeField(auto_now_add=True)

#    class Meta:
#        db_table = 'train_log'

# AI聊天记录模型
class ChatRecord(models.Model):
    message = models.TextField(verbose_name="用户消息")
    reply = models.TextField(verbose_name="AI回复")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="对话时间")

    class Meta:
        ordering = ['-create_time']