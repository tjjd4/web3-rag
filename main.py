import pymupdf4llm

def main():
    print("Hello from web3-rag!")
    md_text = pymupdf4llm.to_markdown("./src/data/How-to-DeFi-Beginner.pdf")
    print(md_text)


if __name__ == "__main__":
    main()
