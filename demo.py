from gensim.models import word2vec
from milvus import Milvus, IndexType, MetricType, Status


def main():
    sentences = word2vec.Text8Corpus("text8")  # 加载语料
    model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=5)  # 训练模型
    word_set = model.wv.index2word  # 单词集合
    word_vec = model.wv.vectors  # word2vec结果向量集合
    milvus = Milvus()
    milvus.connect(host='localhost', port='19530')
    param = {'collection_name': 'word2vec', 'dimension': 200, 'index_file_size': 1024, 'metric_type': MetricType.L2}
    milvus.create_collection(param)

    status, ids = milvus.insert(collection_name='word2vec', records=word_vec)

    # 单词分类
    ivf_param = {'nlist': 100}  # 分成100类
    milvus.create_index('word2vec', IndexType.IVF_FLAT, ivf_param)  # 增加索引
    status, index = milvus.describe_index('word2vec')  # 相当于将word分成100个类别 做了聚类算法

    # 查找相似度最高的单词
    res = milvus.search(collection_name='word2vec', query_records=[list(word_vec[word_set.index('king')])], top_k=10,
                        params={'nprobe': 16})
    for i in range(10):
        id = res[1][0][i].id
        print(word_set[ids.index(id)])

    print(1)


if __name__ == "__main__":
    main()
