document.addEventListener("DOMContentLoaded", () => {
    // 1. Logic cho Menu (Giữ nguyên như cũ)
    const navItems = document.querySelectorAll('.nav-links li a');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // 2. Logic cho Image Slider (Phần mới thêm)
    const bgImg = document.getElementById('bg-img');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    // Mảng chứa tên các file ảnh của bạn
    // BẠN CÓ THỂ THÊM 'img4.jpg', 'img5.jpg'... VÀO ĐÂY NẾU CÓ NHIỀU ẢNH HƠN
    const images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']; 
    
    let currentIndex = 0; // Biến theo dõi vị trí ảnh hiện tại (Bắt đầu từ 0 tương ứng img1.jpg)

    // Hàm cập nhật hình ảnh
    function updateImage() {
        bgImg.src = `img/${images[currentIndex]}`;
    }

    // Xử lý sự kiện click nút Phải (Next)
    nextBtn.addEventListener('click', () => {
        currentIndex++;
        // Nếu đã qua ảnh cuối cùng, quay lại ảnh đầu tiên
        if (currentIndex >= images.length) {
            currentIndex = 0; 
        }
        updateImage();
    });

    // Xử lý sự kiện click nút Trái (Previous)
    prevBtn.addEventListener('click', () => {
        currentIndex--;
        // Nếu lùi quá ảnh đầu tiên, chuyển tới ảnh cuối cùng
        if (currentIndex < 0) {
            currentIndex = images.length - 1; 
        }
        updateImage();
    });

        // Thêm đoạn này vào bên trong DOMContentLoaded
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    // ĐỊA CHỈ NGOK CỦA BẠN (Thay đổi mỗi khi chạy lại Kaggle)
    const KAGGLE_URL = "https://monsoonal-unbolstered-elizabet.ngrok-free.dev/ask";

    async function askAI() {
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Hiển thị tin nhắn người dùng
        chatBox.innerHTML += `<div class="user-msg">${text}</div>`;
        userInput.value = "";
        chatBox.scrollTop = chatBox.scrollHeight;

        // 2. Trạng thái đang xử lý
        const loadingId = "loading-" + Date.now();
        chatBox.innerHTML += `<div class="ai-msg" id="${loadingId}">Đang trả lời...</div>`;

        try {
            const response = await fetch(KAGGLE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            
            // 3. Hiển thị câu trả lời (Xóa dòng loading)
            document.getElementById(loadingId).innerHTML = ` ${data.answer}`;
        } catch (error) {
            document.getElementById(loadingId).innerHTML = "Lỗi kết nối tới Server AI (Kaggle).";
            console.error(error);
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    sendBtn.addEventListener('click', askAI);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askAI();
    });
});

