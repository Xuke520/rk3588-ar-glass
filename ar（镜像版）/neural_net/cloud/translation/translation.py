import argostranslate.package
import argostranslate.translate

class Translation:
    def __init__(self,
                 src_code:str='en',
                 dst_code:str='zh'):
        self.src_code=src_code
        self.dst_code=dst_code

    def translate(self,text:str):
        return argostranslate.translate.translate(text,self.src_code,self.dst_code)



# 安装语言包（如英文→中文）
from_code, to_code = "en", "zh"
# available_packages = argostranslate.package.get_available_packages()
# package = next(p for p in available_packages if p.from_code == from_code and p.to_code == to_code)
# argostranslate.package.install_from_path(package.download())



if __name__=="__main__":
    trans=Translation()
    print(trans.translate("hello world"))