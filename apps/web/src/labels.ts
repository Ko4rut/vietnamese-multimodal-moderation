/** Human-readable translations only; canonical class names come from the API. */
const vietnameseLabels: Record<string, string> = {
  Arts_and_Entertainment: "Nghệ thuật & giải trí",
  Autos_and_Vehicles: "Ô tô & phương tiện",
  Beauty_and_Fitness: "Làm đẹp & thể hình",
  Books_and_Literature: "Sách & văn học",
  Business_and_Industrial: "Kinh doanh & công nghiệp",
  Computers_and_Electronics: "Máy tính & điện tử",
  Finance: "Tài chính",
  Food_and_Drink: "Ẩm thực & đồ uống",
  Games: "Trò chơi",
  Health: "Sức khỏe",
  Hobbies_and_Leisure: "Sở thích & giải trí",
  Home_and_Garden: "Nhà cửa & sân vườn",
  Internet_and_Telecom: "Internet & viễn thông",
  Jobs_and_Education: "Việc làm & giáo dục",
  Law_and_Government: "Pháp luật & chính phủ",
  News: "Tin tức",
  Online_Communities: "Cộng đồng trực tuyến",
  People_and_Society: "Con người & xã hội",
  Pets_and_Animals: "Thú cưng & động vật",
  Real_Estate: "Bất động sản",
  Science: "Khoa học",
  Sensitive_Subjects: "Chủ đề nhạy cảm",
  Shopping: "Mua sắm",
  Sports: "Thể thao",
  Travel_and_Transportation: "Du lịch & giao thông",
};

export function translatedLabel(label: string): string {
  return vietnameseLabels[label] ?? label.replaceAll("_", " ");
}

export function classifierName(classifier: string | null): string {
  if (classifier === "logistic_regression") return "Logistic Regression";
  if (classifier === "linear_svc") return "LinearSVC";
  return "Chưa kết nối";
}
