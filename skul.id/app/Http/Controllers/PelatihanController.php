<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\Pelatihan;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use Maatwebsite\Excel\Facades\Excel;
use App\Exports\PelatihanExport;
use Illuminate\Support\Facades\Auth;
use App\Exports\PesertaPelatihanExport;
use App\Http\Requests\PelatihanStoreRequest;

class PelatihanController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(Request $request)
    {
        $user = User::with('mitraProfile')->find(Auth::id());

        $query = Pelatihan::with(['daftarPelatihan.user'])
            ->withCount('daftarPelatihan')
            ->where('user_id', $user->id)
            ->orderBy('created_at', 'desc');

        // Filter pencarian
        if ($request->filled('judul')) {
            $query->where('nama_pelatihan', 'like', '%' . $request->judul . '%');
        }

        if ($request->filled('kota')) {
            $query->where('kota', 'like', '%' . $request->kota . '%');
        }

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        if ($request->filled('tahun_mulai')) {
            $query->whereYear('tanggal_mulai', $request->tahun_mulai);
        }

        $pelatihanMitraSendiri = $query->paginate(5)->withQueryString()->through(function ($s) {
            $pesertas = collect($s->daftarPelatihan);

            $s->jumlah_asal_sekolah = $pesertas->pluck('asal_sekolah')->unique()->count();
            $s->jumlah_jurusan = $pesertas->pluck('jurusan')->unique()->count();
            $s->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $s->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $s->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();

            return $s;
        });

        // Ambil sertifikasi mitra lain (tanpa pagination)
        $pelatihanMitraLain = Pelatihan::where('user_id', '!=', $user->id)
            ->latest()
            ->limit(6)
            ->get();

        return view('mitra.pelatihan', compact(
            'pelatihanMitraSendiri',
            'pelatihanMitraLain',
            'user'
        ));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(PelatihanStoreRequest $request)
    {
        try {
            if ($request->validated()) {
                $user_id = $request->user()->id;
                $mitra = $request->user();
                $file = $request->file('foto_pelatihan');
                $file_name = $file->hashName(); // nama unik
                $file->move(public_path('storage/assets/pelatihan'), $file_name);

                Pelatihan::create(
                    $request->except('foto_pelatihan') + [
                        'foto_pelatihan' => 'assets/pelatihan/' . $file_name,
                        'user_id' => $user_id,
                        'mitra_id' => $mitra->id,
                    ]
                );
            }

            return redirect()->back()->with('success', 'Pelatihan Berhasil Dibuat');
        } catch (\Exception $e) {
            return redirect()->back()->with('danger', 'Pelatihan Gagal Dibuat');
        }
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, $id)
    {
        $pelatihan = Pelatihan::findOrFail($id);
        $pelatihan->nama_pelatihan = $request->nama_pelatihan;
        $pelatihan->deskripsi = $request->deskripsi;
        $pelatihan->tanggal_mulai = $request->tanggal_mulai;
        $pelatihan->tanggal_selesai = $request->tanggal_selesai;
        $pelatihan->kota = $request->kota;
        $pelatihan->tempat_pelatihan = $request->tempat_pelatihan;
        $pelatihan->biaya = $request->biaya;

        if ($request->status === 'Gratis') {
            $pelatihan->biaya = 0;
            $pelatihan->nomor_rekening = null;
        } else {
            $pelatihan->biaya = $request->biaya;
            $pelatihan->nomor_rekening = $request->nomor_rekening;
        }

        $pelatihan->status = $request->status; // pastikan status disimpan juga

        if ($request->hasFile('foto_pelatihan')) {
            // 🔥 Hapus file lama jika ada
            $oldPath = public_path('storage/' . $pelatihan->foto_pelatihan);
            if (file_exists($oldPath)) {
                unlink($oldPath);
            }

            // 📥 Upload file baru
            $file = $request->file('foto_pelatihan');
            $file_name = $file->hashName();
            $file->move(public_path('storage/assets/pelatihan'), $file_name);

            // 💾 Simpan path baru
            $pelatihan->foto_pelatihan = 'assets/pelatihan/' . $file_name;
        }

        $pelatihan->save();

        return redirect()->back()->with('success', 'Pelatihan berhasil diperbarui.');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $pelatihan = Pelatihan::findOrFail($id);
        if ($pelatihan->user_id != Auth::id()) {
            abort(403);
        }
        $pelatihan->delete();
        return redirect()->back()->with('success', 'Pelatihan berhasil dihapus.');
    }

    public function exportPesertaExcel($id)
    {
        $pelatihan = Pelatihan::findOrFail($id);
        $filename = 'daftar_peserta_' . Str::slug($pelatihan->nama_pelatihan) . '.xlsx';

        return Excel::download(new PesertaPelatihanExport($id), $filename);
    }

    public function exportPelatihan(Request $request)
    {
        return Excel::download(new PelatihanExport($request), 'data_pelatihan.xlsx');
    }
}
