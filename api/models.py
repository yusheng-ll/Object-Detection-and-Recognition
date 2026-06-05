# api/models.py
from django.db import models


class DetectionRecord(models.Model):
    filename = models.CharField(max_length=255, verbose_name='文件名')
    file_path = models.CharField(max_length=500, verbose_name='结果图路径')

    # 🔑 核心指标
    total_count = models.IntegerField(default=0, verbose_name='目标总数')
    density_value = models.FloatField(default=0.0, verbose_name='密度(个/m²)')
    class_data = models.JSONField(default=dict, blank=True, verbose_name='类别统计明细')
    is_abnormal = models.BooleanField(default=False, verbose_name='是否密度异常')

    create_time = models.DateTimeField(auto_now_add=True, verbose_name='检测时间')

    class Meta:
        db_table = 'detection_records'
        verbose_name = '检测记录'
        ordering = ['-create_time']

    def __str__(self):
        return f"{self.filename} - {self.total_count}个目标"

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