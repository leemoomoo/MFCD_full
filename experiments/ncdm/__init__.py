from EduCDM import CDM
class NCDM(CDM):
    def __init__(self,
                 knowledge_n,
                 exer_n,
                 student_n,
                 attn_dim=64,
                 attn_heads=4,
                 attn_drop_ratio=1.0,
                 attn_dropout=0.1):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(
            knowledge_n,
            exer_n,
            student_n,
            attn_dim=attn_dim,
            attn_heads=attn_heads,
            attn_drop_ratio=attn_drop_ratio,
            attn_dropout=attn_dropout
        )

