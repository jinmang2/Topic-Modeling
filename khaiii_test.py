from khaiii import KhaiiiApi


def Analyze(self, text, SEP=' + '):
    res = self.analyze(text)
    f = lambda x: x.__str__().split('\t')[1]
    return SEP.join(list(map(f, res)))


if __name__ == '__main__':
    khai3 = KhaiiiApi()
    setattr(khai3.__class__, 'Analyze', Analyze)
    print(khai3.Analyze('아버지가방에들어가신다.'))
    # 아버지/NNG + 가/JKS + 방/NNG + 에/JKB + 들어가/VV + 시/EP + ㄴ다/EF + ./SF