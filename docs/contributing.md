# 🤝 Awesome-Cloud 投稿与协作指南

欢迎参与 Awesome-Cloud 项目的维护！为了确保周刊内容的质量，确保存档工作的规范化，请大家在提交汇报内容（Pull Request）前，严格遵循以下 SOP（标准作业程序）。

## 📋 投稿完整流程 (SOP)

### 第一步：准备工作 (Fork & Clone)
1.  **Fork 仓库**：在页面右上角点击 `Fork` 按钮，将仓库复制到你自己的 GitHub 账号下。
2.  **Clone 到本地**：将你账号下 **Fork 后的仓库**克隆到本地电脑。
    * 注意：是 Clone 你自己的仓库，不是课题组的仓库，否则无法推送。
    ```bash
    # 请将 <你的用户名> 替换为你实际的 GitHub Username
    git clone https://github.com/<你的用户名>/Awesome-Cloud.git
    ```
3.  **新建分支**：**禁止**直接在 main 分支修改。请新建一个分支，命名格式为 `weekly-issue-xx`（xx 为期数）。
    ```bash
    # 例如：第 25 期
    git checkout -b weekly-issue-25
    ```

### 第二步：添加内容（核心环节）
请将你的汇报内容整理为 Markdown 格式，并严格遵守以下文件结构：

1.  **上传图片（命名规范）**：
    * 将图片保存到项目根目录的 `images/` 文件夹中。
    * **命名格式**：`issue-xx(期数)-xx(图片编号)`。
    * ✅ 正确示例：`issue-25-01.png`, `issue-25-02.jpg`
2.  **创建文档**：
    * 在 `docs/` 目录下新建 Markdown 文件，命名格式为 `issue-xx.md`。
    * **⚠️ 重点：图片引用路径**
        * 请务必使用**相对路径**引用图片，否则网页预览会裂图。
        * ✅ 正确写法：`![](../images/issue-25-01.png)`
        * ❌ 错误写法：`![](images/issue-25-01.png)` 或绝对路径

### 第三步：更新索引（必须执行！）
上传完正文后，**必须**手动更新以下两个目录文件，否则大家找不到你的文章：

1.  **按时间索引**：
    * 打开 `docs/weekly.md`。
    * 在列表顶部增加一行，格式：`- [第 XX 期：<主题>](./issue-xx.md) (<日期>)`。
2.  **按话题索引**：
    * 打开根目录的 `README.md`。
    * 找到“**已讨论话题索引**”部分，将你的文章分类添加到对应的板块下。

### 第四步：提交与推送
提交信息（Commit Message）请保持统一格式。

```bash
git add .
# 格式：docs: release issue xx
git commit -m "docs: release issue 25"
git push origin weekly-issue-25
```

### 第五步：发起 Pull Request (PR)

1.  回到 GitHub 原仓库页面，你会看到黄色的提示条 "Compare & pull request"。
2.  点击按钮，进入 PR 页面。
3.  **Self-Check（提交前自查）**：
      * [ ] 分支名是否为 `weekly-issue-xx`？
      * [ ] 图片是否已按 `issue-xx-xx` 规范命名？
      * [ ] `docs/weekly.md` 和 `README.md` 索引是否已更新？
      * [ ] **关键**：点开 Files changed 预览，确认图片能正常显示？
4.  确认无误后，点击 **Create pull request**，并通知 Maintainer进行 Review。

-----

## 🛠 常见问题排查

  * **图片不显示？**
      * 检查是否使用了 `../images/` 开头的相对路径。
  * **Commit 之后发现有错别字？**
      * 直接在本地修改文件，再次 `git add` 和 `git commit`，然后 `git push` 即可，PR 会自动更新，**无需**关闭 PR 重新提交。

感谢大家为 Awesome-Cloud 添砖加瓦！🚀