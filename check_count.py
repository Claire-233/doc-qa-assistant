import traceback
print("脚本开始运行...", flush=True)

try:
    import chromadb
    print("chromadb 导入成功", flush=True)
    
    client = chromadb.PersistentClient(path="./storage/chroma")
    print("客户端连接成功", flush=True)
    
    # 列出当前所有的集合
    cols = client.list_collections()
    print(f"当前数据库内的集合: {cols}", flush=True)
    
    # 尝试获取 doc_qa 集合
    try:
        collection = client.get_collection("doc_qa")
        print(f"库内共 {collection.count()} 条", flush=True)
    except Exception as e:
        print(f"获取 'doc_qa' 集合失败: {e}", flush=True)
        print("提示：集合名称可能不匹配，请检查上面的集合列表", flush=True)

except Exception as e:
    print("发生异常：", flush=True)
    traceback.print_exc()